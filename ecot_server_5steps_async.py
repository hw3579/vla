"""
轻量级服务器实现，用于通过REST API部署Embodied-CoT模型。
支持5步异步处理模式，提高推理效率。
使用改良版坐标处理机制。

使用方法:
python ecot_server_5steps_async_improved.py --port 8000

依赖:
    => OpenVLA/ECot依赖 
    => `pip install uvicorn fastapi json-numpy PIL numpy opencv-python`

客户端示例 (假设服务器运行在 0.0.0.0:8000):

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
        "instruction": "place the watermelon on the towel",
        "session_id": "unique_session_identifier"  # 使用会话ID跟踪对话状态
    }
).json()

# 获取结果
action = response
"""

import datetime
import uuid
import os
import json
import logging
import traceback
import textwrap
import sys
import time
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

# 从utils.tools导入改良版函数
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


# === PromptManager5stepAsync ===
class PromptManager5stepAsync:
    """
    结合5步执行逻辑和异步批处理的提示管理器。
    每5步更新一次高层计划，支持批量历史更新。
    """
    def __init__(self, history_adaptive=False):
        self.history = ""
        self.update_counter = 0  # 步骤计数器
        self.highlevel_frequency = 5  # 每5步更新一次高层计划
        self.history_adaptive = history_adaptive
        
    def update_history(self, text):
        """更新历史记录"""
        self.history = text
        self.update_counter += 1
        
    def batch_update_history(self, texts):
        """批量更新历史记录，保留最后一个文本"""
        if texts:
            self.history = texts[-1]
            self.update_counter += 1
    
    def generate_prompts(self, task_description):
        """
        根据当前步骤生成适当的提示。
        每5步（highlevel_frequency）重新生成完整计划，
        其余步骤只处理当前执行部分。
        """
        SYSTEM_PROMPT = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        
        prompts = []
        # 第一步或每5步一次，生成完整提示（包括高层计划）
        if self.update_counter == 0 or self.update_counter % self.highlevel_frequency == 0:
            prompts.append(f"{SYSTEM_PROMPT} USER: What action should the robot take to {task_description.lower()}? ASSISTANT: TASK:")
        else:
            # 其他步骤，只生成执行部分
            if self.history:
                # 从历史中提取任务和计划部分
                task_plan = ""
                if "TASK:" in self.history and "PLAN:" in self.history:
                    task_start = self.history.find("TASK:")
                    plan_start = self.history.find("PLAN:")
                    visible_objects_start = self.history.find("VISIBLE OBJECTS:")
                    
                    if visible_objects_start > plan_start > task_start:
                        task_plan = self.history[task_start:visible_objects_start]
                
                # 生成集中于当前执行步骤的提示
                prompts.append(f"{SYSTEM_PROMPT} USER: What action should the robot take to {task_description.lower()}? ASSISTANT: {task_plan}VISIBLE OBJECTS:")
        
        return prompts


# === 服务器接口 ===
class ECotServerAsync:
    def __init__(
        self,
        model_path: Union[str, Path],
        attn_implementation: Optional[str] = "flash_attention_2",
        save_files: bool = True,
        history_adaptive: bool = False,
    ):
        """
        ECot异步模型服务器；提供'/act'接口来预测给定图像+指令的动作。
            => 输入 {"image": np.ndarray, "instruction": str, "session_id": str}
            => 返回 np.ndarray 动作数组（7个元素）
        """
        self.model_path = model_path
        self.attn_implementation = attn_implementation
        self.save_files = save_files
        self.history_adaptive = history_adaptive
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tags = [f" {tag}" for tag in get_cot_tags_list()]
        
        # 用于存储会话状态的字典
        self.sessions = {}
        
        # 跟踪GPU使用情况
        self.last_inference_time = time.time()

        # 加载模型
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

    def get_prompt(self, instruction: str, session_id: str = None) -> str:
        """生成适用于ECot模型的提示"""
        # 如果没有会话ID，使用默认提示
        if session_id is None:
            SYSTEM_PROMPT = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
            return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
        
        # 为会话创建提示管理器（如果不存在）
        if session_id not in self.sessions:
            self.sessions[session_id] = PromptManager5stepAsync(history_adaptive=self.history_adaptive)
            
        # 使用该会话的提示管理器生成提示
        prompts = self.sessions[session_id].generate_prompts(instruction)
        return prompts[0]
    
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
            
            # 检查GPU利用率，必要时等待
            current_time = time.time()
            time_since_last = current_time - self.last_inference_time
            if time_since_last < 0.1:  # 如果上次推理是在100ms以内
                torch.cuda.empty_cache()  # 清理GPU内存
                
            # 添加推理时间记录
            start_time = time.time()
                
            # 运行模型推理
            prompt = self.get_prompt(instruction, session_id)
            torch.manual_seed(seed)
            
            try:
                inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
                
                # 根据是否为首次调用决定最大新令牌数
                max_tokens = 1024 if session_id not in self.sessions or self.sessions[session_id].update_counter == 0 else 60
                
                action, generated_ids = self.model.predict_action(
                    **inputs, 
                    unnorm_key=unnorm_key, 
                    do_sample=False, 
                    max_new_tokens=max_tokens
                )
                
                # 记录当前推理时间
                self.last_inference_time = time.time()
                
                # 计算推理时间
                inference_time = self.last_inference_time - start_time
                logging.info(f"推理时间: {inference_time:.4f}秒")
                
                generated_texts = self.processor.batch_decode(generated_ids)
                generated_text = generated_texts[-1]
                
                # 更新会话状态
                if session_id is not None and session_id in self.sessions:
                    if len(generated_texts) > 1:
                        # 使用批量更新
                        self.sessions[session_id].batch_update_history(generated_texts)
                    else:
                        # 使用单个更新
                        self.sessions[session_id].update_history(generated_text)
                    
            except Exception as e:
                logging.error(traceback.format_exc())
                raise HTTPException(
                    status_code=400, 
                    detail=f"模型处理错误: {str(e)}"
                )

            # 获取步骤计数器
            step_counter = None
            if session_id is not None and session_id in self.sessions:
                step_counter = self.sessions[session_id].update_counter

            # 创建可视化推理图像并保存数据
            try:
                reasoning_image, metadata = create_reasoning_image(image, generated_text, self.tags)
                
                # 将步骤信息加入metadata
                if step_counter is not None:
                    metadata["step_counter"] = step_counter
                    
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
            
            # 只返回动作数组（7个值）
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
                    "{'image': np.ndarray, 'instruction': str, 'session_id': str(可选)}\n"
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
                "mode": "5step-async-improved"
            }

        # 添加动作预测端点
        self.app.post("/act")(self.predict_action)
        
        # 添加会话管理端点
        @self.app.post("/reset_session")
        async def reset_session(data: Dict[str, str]):
            if "session_id" not in data:
                raise HTTPException(status_code=400, detail="需要session_id参数")
                
            session_id = data["session_id"]
            if session_id in self.sessions:
                del self.sessions[session_id]
                return {"status": "success", "message": f"会话 {session_id} 已重置"}
            else:
                return {"status": "success", "message": f"会话 {session_id} 不存在或已被重置"}

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
    history_adaptive: bool = False                  # 是否使用自适应历史长度


@draccus.wrap()
def run_server(cfg: ServerConfig) -> None:
    """启动ECot异步服务器"""

    # 检查predictions目录
    if cfg.save_files:
        check_predictions_directory()
    
    server = ECotServerAsync(
        cfg.model_path, 
        save_files=cfg.save_files,
        history_adaptive=cfg.history_adaptive
    )
    
    logging.info("5step-async模式已启用 - 异步处理并每5步更新一次高层计划")
    
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    run_server()