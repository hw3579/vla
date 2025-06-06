"""
轻量级服务器实现，用于通过REST API部署Embodied-CoT模型。
支持多种量化方式以优化内存和速度。

使用方法:
```
python ecot_server_quantization.py --port 8000 --quantization 4bit
```

依赖:
    => OpenVLA/ECot依赖 
    => `pip install uvicorn fastapi json-numpy PIL numpy opencv-python bitsandbytes`

客户端示例 (假设服务器运行在 0.0.0.0:8000):

```
import requests
import json_numpy
import numpy as np
from PIL import Image

json_numpy.patch()

# 读取图像
image = np.array(Image.open("test_obs.png"))

# 发送请求
response = requests.post(
    "http://0.0.0.0:8000/act",
    json={
        "image": image, 
        "instruction": "place the watermelon on the towel"
    }
).json()

# 获取结果
action = response  # 直接返回7个元素的numpy数组
```
"""

import datetime
import uuid
import os
import json
import logging
import traceback
import enum
import textwrap
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
import time

import draccus
import torch
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# 导入JSON-Numpy扩展以支持numpy数组的JSON序列化
import json_numpy
json_numpy.patch()

# 导入改良版工具函数
from utils.tools import (
    CotTag,
    get_cot_tags_list,
    name_to_random_color,
    draw_gripper,
    draw_bboxes,
    get_metadata,
    split_reasoning,
    create_reasoning_image,
    check_predictions_directory,
    save_prediction_data
)


# === 服务器接口 ===
class ECotServerQuantization:
    def __init__(
        self,
        model_path: Union[str, Path],
        attn_implementation: Optional[str] = "flash_attention_2",
        save_files: bool = True,
        quantization: str = "none",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        ECot模型服务器，支持量化；提供'/act'接口来预测给定图像+指令的动作。
            => 输入 {"image": np.ndarray, "instruction": str}
            => 返回 np.ndarray（7个元素的动作数组）
            
        Args:
            model_path: 模型路径
            attn_implementation: 注意力实现方式 ("eager", "sdpa", "flash_attention_2")
            save_files: 是否保存预测数据
            quantization: 量化方式 ("none", "4bit", "8bit")
            load_in_8bit: 是否使用8位量化加载模型
            load_in_4bit: 是否使用4位量化加载模型
        """
        self.model_path = model_path
        self.attn_implementation = attn_implementation
        self.save_files = save_files
        self.quantization = quantization
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # 确保不同时使用8位和4位量化
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("不能同时使用8位和4位量化!")
            
        # 根据quantization参数设置load_in_8bit和load_in_4bit
        if self.quantization == "4bit":
            self.load_in_4bit = True
            self.load_in_8bit = False
        elif self.quantization == "8bit":
            self.load_in_8bit = True
            self.load_in_4bit = False
            
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tags = [f" {tag}" for tag in get_cot_tags_list()]

        # 记录量化状态
        self.quantization_mode = "None"
        if self.load_in_4bit:
            self.quantization_mode = "4-bit"
        elif self.load_in_8bit:
            self.quantization_mode = "8-bit"
            
        logging.info(f"使用 {self.quantization_mode} 量化模式加载模型...")

        # 加载模型
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
        # 根据量化设置加载模型
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        ).to(self.device)
        
        logging.info(f"模型已成功加载: {model_path}")
        
        # 打印模型内存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"GPU内存使用: 分配 {memory_allocated:.2f} GB, 保留 {memory_reserved:.2f} GB")

    def get_prompt(self, instruction: str) -> str:
        """生成适用于ECot模型的提示"""
        SYSTEM_PROMPT = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"


    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            # 支持double_encode情况
            if double_encode := "encoded" in payload:
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # 解析payload
            if "image" not in payload or "instruction" not in payload:
                raise HTTPException(
                    status_code=400,
                    detail="缺少必要字段: image和instruction",
                )

            image_array, instruction = payload["image"], payload["instruction"]
            image = Image.fromarray(np.array(image_array, dtype=np.uint8)).convert("RGB")
            unnorm_key = payload.get("unnorm_key", "bridge_orig")
            seed = payload.get("seed", 0)

            # 运行模型推理
            prompt = self.get_prompt(instruction)
            torch.manual_seed(seed)
            
            try:
                # 记录推理开始时间
                start_time = time.time()
                
                inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
                action, generated_ids = self.model.predict_action(
                    **inputs, 
                    unnorm_key=unnorm_key, 
                    do_sample=False, 
                    max_new_tokens=1024
                )
                
                # 计算推理时间
                inference_time = time.time() - start_time
                logging.info(f"推理时间: {inference_time:.4f}秒")
                
                generated_text = self.processor.batch_decode(generated_ids)[0]
            except Exception as e:
                logging.error(traceback.format_exc())
                raise HTTPException(
                    status_code=400, 
                    detail=f"模型处理错误: {str(e)}"
                )

            # 创建可视化推理图像
            try:
                reasoning_image, metadata = create_reasoning_image(image, generated_text, self.tags)
                
                # 保存数据到文件夹
                if self.save_files:
                    save_dir = save_prediction_data(
                        image, 
                        action, 
                        reasoning_image, 
                        generated_text, 
                        metadata,
                        instruction,
                        inference_time  # 添加推理时间参数
                    )
                    logging.info(f"预测数据已保存到: {save_dir}")
                
            except Exception as e:
                logging.error(f"创建或保存推理数据时出错: {str(e)}")
                logging.error(traceback.format_exc())
                # 如果可视化处理失败，仍然返回动作坐标
                return JSONResponse(action.tolist() if hasattr(action, "tolist") else action)
            
            # 仅返回动作坐标
            result = action.tolist() if hasattr(action, "tolist") else action

            if double_encode:
                return JSONResponse(json_numpy.dumps(result))
            else:
                return JSONResponse(result)
                
        except HTTPException:
            raise
        except Exception as e:
            logging.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=(
                    f"服务器错误: {str(e)}\n"
                    "请确保您的请求符合预期格式:\n"
                    "{'image': np.ndarray, 'instruction': str}\n"
                    "可选参数: 'unnorm_key': str, 'seed': int"
                ),
            )

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()

        # 添加CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 添加健康检查端点
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy", 
                "model": str(self.model_path),
                "quantization": self.quantization_mode
            }

        # 添加动作预测端点
        self.app.post("/act")(self.predict_action)

        # 配置服务器，增加超时和请求大小限制
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            timeout_keep_alive=120,
            limit_concurrency=2,
        )
        server = uvicorn.Server(config)
        server.run()


@dataclass
class ServerConfig:
    # 基本配置
    model_path: Union[str, Path] = "Embodied-CoT/ecot-openvla-7b-bridge"  # 模型路径
    
    # 服务器配置
    host: str = "0.0.0.0"                           # 主机IP地址
    port: int = 8000                                # 主机端口
    save_files: bool = True                         # 是否保存预测数据
    
    # 量化配置
    quantization: str = "none"                      # 量化方式 ("none", "4bit", "8bit")
    load_in_8bit: bool = False                      # 是否使用8位量化
    load_in_4bit: bool = False                      # 是否使用4位量化


@draccus.wrap()
def run_server(cfg: ServerConfig) -> None:
    """启动ECot量化版服务器"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 检查predictions目录
    if cfg.save_files:
        check_predictions_directory()
    
    # 检查量化参数
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("不能同时使用8位和4位量化!")
    
    if cfg.quantization not in ["none", "4bit", "8bit"]:
        raise ValueError(f"不支持的量化方式: {cfg.quantization}，支持的选项: none, 4bit, 8bit")
    
    # 显示启动信息
    quant_mode = "none"
    if cfg.quantization == "4bit" or cfg.load_in_4bit:
        quant_mode = "4-bit"
    elif cfg.quantization == "8bit" or cfg.load_in_8bit:
        quant_mode = "8-bit"
    
    logging.info(f"启动ECot量化版服务器，量化模式: {quant_mode}")
    
    server = ECotServerQuantization(
        cfg.model_path, 
        save_files=cfg.save_files,
        quantization=cfg.quantization,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit
    )
    
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    run_server()