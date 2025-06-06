"""
轻量级服务器实现，用于通过REST API部署Embodied-CoT模型。

使用方法:
```
python ecot_server.py --port 8000
```

依赖:
    => OpenVLA/ECot依赖 
    => `pip install uvicorn fastapi json-numpy PIL numpy opencv-python`

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
action = response["action"]
reasoning_image = np.array(response["reasoning_image"])
```
"""

import time
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


from utils.steps5 import hf_to_vllm, PromptManager5step
from utils.tools import (
    get_cot_tags_list, 
    create_reasoning_image, 
    check_predictions_directory, 
    save_prediction_data,
    split_reasoning,
    name_to_random_color,
    get_metadata
)







# === 服务器接口 ===
class ECotServer:
    def __init__(
        self,
        model_path: Union[str, Path],
        attn_implementation: Optional[str] = "flash_attention_2",
        save_files: bool = True,
        use_5step: bool = True,  # 新增参数，控制是否使用5step模式
    ):
        """
        ECot模型服务器；提供'/act'接口来预测给定图像+指令的动作。
            => 输入 {"image": np.ndarray, "instruction": str}
            => 返回 {"action": np.ndarray, "reasoning_image": np.ndarray, "generated_text": str}
        """
        self.model_path = model_path
        self.attn_implementation = attn_implementation
        self.save_files = save_files
        self.use_5step = use_5step  # 是否使用5step模式
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tags = [f" {tag}" for tag in get_cot_tags_list()]
        
        # 初始化5step提示管理器
        self.prompt_manager = PromptManager5step() if use_5step else None
        
        # 用于存储会话状态的字典
        self.sessions = {}

        # 加载模型
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # load_in_4bit=True,
        ).to(self.device)

    def get_prompt(self, instruction: str, session_id: str = None) -> str:
        """生成适用于ECot模型的提示"""
        SYSTEM_PROMPT = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        
        # 使用5step模式生成提示
        if self.use_5step and session_id is not None:
            # 为会话创建提示管理器（如果不存在）
            if session_id not in self.sessions:
                self.sessions[session_id] = PromptManager5step()
                
            # 使用该会话的提示管理器生成提示
            prompts = self.sessions[session_id].generate_prompts(instruction)
            return prompts[0]
        else:
            # 使用常规模式生成提示
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
            
            # 获取会话ID（如果有）
            session_id = payload.get("session_id", None)
            
            # 添加推理时间记录
            start_time = time.time()
            
            # 运行模型推理
            prompt = self.get_prompt(instruction, session_id)
            torch.manual_seed(seed)
            
            try:
                inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
                action, generated_ids = self.model.predict_action(
                    **inputs, 
                    unnorm_key=unnorm_key, 
                    do_sample=False, 
                    max_new_tokens=1024
                )
                generated_text = self.processor.batch_decode(generated_ids)[0]
                
                # 计算推理时间
                inference_time = time.time() - start_time
                logging.info(f"推理时间: {inference_time:.4f}秒")
                
                # 如果使用5step模式，更新会话状态
                if self.use_5step and session_id is not None and session_id in self.sessions:
                    self.sessions[session_id].update_history(generated_text)
                    
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
                
                # 如果使用5step模式，将步骤计数器信息添加到metadata中保存
                if self.use_5step and session_id is not None and session_id in self.sessions:
                    step_info = {
                        "step_counter": self.sessions[session_id].update_counter,
                        "high_level_frequency": self.sessions[session_id].highlevel_frequency
                    }
                    # 将步骤信息添加到metadata.json文件
                    metadata_path = os.path.join(save_dir, "metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            meta_data = json.load(f)
                        meta_data.update(step_info)
                        with open(metadata_path, "w") as f:
                            json.dump(meta_data, f, indent=2)
                
            except Exception as e:
                logging.error(f"创建或保存推理数据时出错: {str(e)}")
                logging.error(traceback.format_exc())
                # 如果可视化处理失败，仍然返回动作坐标
                return JSONResponse(action.tolist() if hasattr(action, "tolist") else action)
            
            # 只返回动作坐标，与原始版本一致
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
                    "可选参数: 'unnorm_key': str, 'seed': int, 'session_id': str"
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
            return {"status": "healthy", "model": str(self.model_path)}

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
    host: str = "0.0.0.0"                                               # 主机IP地址
    port: int = 8000                                                    # 主机端口
    save_files: bool = True                                          # 是否保存预测数据
    use_5step: bool = True                        # 是否使用5step模式



@draccus.wrap()
def run_server(cfg: ServerConfig) -> None:
    """启动ECot服务器"""

    # 检查predictions目录
    if cfg.save_files:
        check_predictions_directory()
    
    server = ECotServer(
        cfg.model_path, 
        save_files=cfg.save_files,
        use_5step=cfg.use_5step
    )
    
    if cfg.use_5step:
        logging.info("5step模式已启用 - 每5步更新一次高层计划")
    
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    run_server()