
"""
轻量级服务器实现，用于通过REST API部署Embodied-CoT模型。
包含faithfulness评估功能，可测量模型推理一致性。

使用方法:
```
python ecot_server_faith.py --port 8000
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
        "instruction": "place the watermelon on the towel",
        "session_id": "unique_session_identifier"  # 使用会话ID跟踪对话状态
    }
).json()

# 获取结果
action = response["action"]
faith_scores = response["faith_scores"]  # 8个faithfulness分数
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


# === PromptManager 类，支持faithfulness评估 ===
class PromptManagerFaith:
    """用于管理会话中的提示和历史记录的管理器，包含faithfulness评估功能"""
    def __init__(self, history_adaptive=False):
        self.history = ""
        self.history_adaptive = history_adaptive
        
    def update_history(self, text, idx=None):
        """更新历史记录"""
        self.history = text
        
    def batch_update_history(self, texts):
        """批量更新历史记录，使用最后一个文本作为当前历史"""
        if texts:
            self.history = texts[-1]
    
    def generate_prompts(self, task_description):
        """根据任务描述和当前历史记录生成提示"""
        SYSTEM_PROMPT = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        
        prompts = []
        if not self.history:
            # 首次生成提示
            prompts.append(f"{SYSTEM_PROMPT} USER: What action should the robot take to {task_description.lower()}? ASSISTANT: TASK:")
        else:
            # 从历史中提取任务和计划部分
            task_plan = ""
            if "TASK:" in self.history and "PLAN:" in self.history:
                task_start = self.history.find("TASK:")
                plan_start = self.history.find("PLAN:")
                visible_objects_start = self.history.find("VISIBLE OBJECTS:")
                
                if visible_objects_start > plan_start > task_start:
                    task_plan = self.history[task_start:visible_objects_start]
            
            # 生成后续提示
            prompts.append(f"{SYSTEM_PROMPT} USER: What action should the robot take to {task_description.lower()}? ASSISTANT: {task_plan}VISIBLE OBJECTS:")
        
        return prompts
    
    def generate_prompts_faith(self, task_description):
        """生成用于测试faithfulness的提示，包含8个不同的变体"""
        SYSTEM_PROMPT = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        
        faith_prompts = []
        
        # 如果没有历史记录，无法进行faithfulness测试
        if not self.history:
            return faith_prompts
        
        # 从历史中提取任务、计划和可见对象部分
        task_plan = ""
        if "TASK:" in self.history and "PLAN:" in self.history:
            task_start = self.history.find("TASK:")
            plan_start = self.history.find("PLAN:")
            visible_objects_start = self.history.find("VISIBLE OBJECTS:")
            
            if visible_objects_start > plan_start > task_start:
                task_plan = self.history[task_start:visible_objects_start]
        
        # 创建8个不同的提示变体进行测试
        for i in range(8):
            # 这里我们使用ACTION前缀直接跳到动作生成部分
            faith_prompts.append(f"{SYSTEM_PROMPT} USER: What action should the robot take to {task_description.lower()}? ASSISTANT: {task_plan}VISIBLE OBJECTS:{self.history.split('VISIBLE OBJECTS:')[1].split('ACTION:')[0]}ACTION:")
        
        return faith_prompts


# === 实用工具函数 ===
def check_predictions_directory(directory="predictions", warn_if_non_empty=True):
    """检查predictions目录是否存在且是否非空"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return False
    
    items = os.listdir(directory)
    is_non_empty = len(items) > 0
    
    if is_non_empty and warn_if_non_empty:
        print("\033[93m警告: 'predictions'目录非空，包含", len(items), "个项目。")
        print("如果继续，新的预测结果将添加到此目录中。")
        print("是否继续? [y/N]\033[0m")
        
        response = input().strip().lower()
        if response != 'y' and response != 'yes':
            print("已取消启动服务器。")
            sys.exit(0)
    
    return is_non_empty


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"

def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]

def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
            else:
                new_parts[k] = v

    return new_parts

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]

def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_gripper(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        try:
            gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
            gripper_pos = gripper_pos.split("[")[-1]
            gripper_pos = gripper_pos.split("]")[0]
            # 确保不处理空字符串
            gripper_pos = [int(x.strip()) for x in gripper_pos.split(",") if x.strip()]
            # 确保有足够的坐标点
            if len(gripper_pos) >= 2 and len(gripper_pos) % 2 == 0:
                gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
                metadata["gripper"] = gripper_pos
        except (ValueError, IndexError) as e:
            logging.warning(f"解析抓取位置时出错: {str(e)}，使用默认值")

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            try:
                obj = sample.split("[")[0]
                if obj == "" or "[" not in sample:
                    continue
                    
                # 过滤掉空字符串并转换为整数
                coords_str = sample.split("[")[-1].split(",")
                coords = [int(n.strip()) for n in coords_str if n.strip()]
                
                # 确保坐标数量正确
                if len(coords) == 4:
                    metadata["bboxes"][obj.strip()] = coords
            except (ValueError, IndexError) as e:
                logging.warning(f"解析边界框时出错: {str(e)}, sample: {sample}")
                continue

    return metadata

def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name
        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

def create_reasoning_image(image, generated_text, tags):
    """生成带有推理过程可视化的图像"""
    reasoning = split_reasoning(generated_text, tags)
    text = [tag + reasoning[tag] for tag in [' TASK:',' PLAN:',' SUBTASK REASONING:',' SUBTASK:',
                                           ' MOVE REASONING:',' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:'] if tag in reasoning]
    metadata = get_metadata(reasoning)
    bboxes = {}
    for k, v in metadata["bboxes"].items():
        if k[0] == ",":
            k = k[1:]
        bboxes[k.lstrip().rstrip()] = v

    caption = ""
    for t in text:
        wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False) 
        word_list = wrapper.wrap(text=t) 
        caption_new = ''
        for ii in word_list[:-1]:
            caption_new = caption_new + ii + '\n      '
        caption_new += word_list[-1]

        caption += caption_new.lstrip() + "\n\n"

    base = Image.fromarray(np.ones((480, 640, 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except IOError:
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
            logging.warning("无法加载指定字体，使用默认字体")

    color = (0,0,0)
    draw.text((30, 30), caption, color, font=font)

    img_arr = np.array(image)
    draw_gripper(img_arr, metadata["gripper"])
    draw_bboxes(img_arr, bboxes)

    text_arr = np.array(base)

    if img_arr.shape[0] != text_arr.shape[0]:
        img_arr = cv2.resize(img_arr, (text_arr.shape[1], text_arr.shape[0]))

    reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))
    return reasoning_img, metadata

def save_prediction_data(image, action, reasoning_image, generated_text, metadata, instruction, faith_scores=None, faith_actions=None):
    """保存预测数据到文件夹，包含faithfulness评估结果"""
    # 创建用于保存数据的目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = str(uuid.uuid4())[:8]
    save_dir = os.path.join("predictions", f"{timestamp}_{session_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存原始图像
    image_path = os.path.join(save_dir, "input_image.png")
    image.save(image_path)
    
    # 保存推理图像
    reasoning_image_path = os.path.join(save_dir, "reasoning_image.png")
    reasoning_image.save(reasoning_image_path)
    
    # 保存生成的文本
    text_path = os.path.join(save_dir, "generated_text.txt")
    with open(text_path, "w") as f:
        f.write(f"Instruction: {instruction}\n\n")
        f.write(generated_text)
    
    # 保存元数据、动作和faithfulness分数
    metadata_path = os.path.join(save_dir, "metadata.json")
    save_data = {
        "action": action.tolist() if hasattr(action, "tolist") else action,
        "metadata": metadata,
        "timestamp": timestamp,
        "instruction": instruction
    }
    
    # 如果有faithfulness评估结果，添加到元数据
    if faith_scores is not None:
        save_data["faith_scores"] = faith_scores.tolist() if hasattr(faith_scores, "tolist") else faith_scores
    
    if faith_actions is not None:
        save_data["faith_actions"] = faith_actions.tolist() if hasattr(faith_actions, "tolist") else faith_actions
        
    with open(metadata_path, "w") as f:
        json.dump(save_data, f, indent=2)
    
    # 如果有faithfulness动作，保存为NumPy数组
    if faith_actions is not None:
        faith_actions_path = os.path.join(save_dir, "faith_actions.npy")
        with open(faith_actions_path, "wb") as f:
            np.save(f, faith_actions)
    
    return save_dir


# === 服务器接口 ===
class ECotServerFaith:
    def __init__(
        self,
        model_path: Union[str, Path],
        attn_implementation: Optional[str] = "flash_attention_2",
        save_files: bool = True,
        history_adaptive: bool = False,
    ):
        """
        ECot服务器 - 包含faithfulness评估；提供'/act'接口来预测给定图像+指令的动作。
            => 输入 {"image": np.ndarray, "instruction": str, "session_id": str}
            => 返回 {"action": np.ndarray, "faith_scores": np.ndarray}
        """
        self.model_path = model_path
        self.attn_implementation = attn_implementation
        self.save_files = save_files
        self.history_adaptive = history_adaptive
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tags = [f" {tag}" for tag in get_cot_tags_list()]
        
        # 用于存储会话状态的字典
        self.sessions = {}
        
        # 会话步骤计数
        self.session_steps = {}
        
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

        logging.info(f"已成功加载模型: {model_path}")
        logging.info(f"Faithfulness评估模式已启用，history_adaptive={history_adaptive}")

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
                
            # 跟踪步骤数
            step_num = None
            if session_id:
                if session_id not in self.session_steps:
                    self.session_steps[session_id] = 0
                else:
                    self.session_steps[session_id] += 1
                step_num = self.session_steps[session_id]
                
            # 为新会话创建PromptManager
            if session_id and session_id not in self.sessions:
                self.sessions[session_id] = PromptManagerFaith(history_adaptive=self.history_adaptive)
            
            # 根据会话状态决定生成策略
            is_first_step = False
            if session_id and session_id in self.sessions:
                # 非首次请求，使用会话历史
                prompts = self.sessions[session_id].generate_prompts(instruction)
                prompt = prompts[0]
                max_new_tokens = 60  # 后续请求使用较小的token数
            else:
                # 首次请求或无会话
                is_first_step = True
                SYSTEM_PROMPT = (
                    "A chat between a curious user and an artificial intelligence assistant. "
                    "The assistant gives helpful, detailed, and polite answers to the user's questions."
                )
                prompt = f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
                max_new_tokens = 1024  # 首次请求生成完整推理
            
            # 运行模型推理
            torch.manual_seed(seed)
            start_time = time.time()
            
            try:
                inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
                
                # 主要动作生成
                action, generated_ids = self.model.predict_action(
                    **inputs, 
                    unnorm_key=unnorm_key, 
                    do_sample=False, 
                    max_new_tokens=max_new_tokens
                )
                
                # 解码生成的文本
                generated_texts = self.processor.batch_decode(generated_ids)
                generated_text = generated_texts[-1]
                
                # 更新会话状态
                if session_id and session_id in self.sessions:
                    if len(generated_texts) > 1:
                        # 使用批量更新
                        self.sessions[session_id].batch_update_history(generated_texts)
                    else:
                        # 使用单个更新
                        self.sessions[session_id].update_history(generated_text)
                
                # Faithfulness评估 - 仅在已有历史记录时执行
                faith_actions = None
                faith_scores = None
                
                if session_id and session_id in self.sessions and not is_first_step:
                    # 生成用于faithfulness测试的提示
                    faith_prompts = self.sessions[session_id].generate_prompts_faith(instruction)
                    
                    if faith_prompts:
                        # 获取faithfulness测试的动作
                        faith_inputs = []
                        for faith_prompt in faith_prompts:
                            faith_inputs.append(self.processor(faith_prompt, image).to(self.device, dtype=torch.bfloat16))
                        
                        # 批量处理所有faith输入
                        faith_batch_actions = []
                        for faith_input in faith_inputs:
                            with torch.no_grad():
                                _, faith_action = self.model.forward(
                                    **faith_input,
                                    max_new_tokens=8
                                )
                                faith_action = self.model.postprocess_action(faith_action, unnorm_key=unnorm_key)
                                faith_batch_actions.append(faith_action)
                        
                        # 合并所有faith动作
                        faith_actions = torch.stack(faith_batch_actions, dim=0).cpu().numpy()
                        
                        # 计算与主动作的差异
                        action_np = action.cpu().numpy()
                        combined_actions = np.concatenate([faith_actions, np.expand_dims(action_np, 0)], axis=0)
                        faith_scores = np.sum(np.abs(faith_actions - action_np), axis=1)
                        
                        logging.info(f"Faith scores: {faith_scores}")
                
                inference_time = time.time() - start_time
                logging.info(f"推理时间: {inference_time:.4f}秒")
                
                # 记录当前推理时间
                self.last_inference_time = time.time()
                
            except Exception as e:
                logging.error(traceback.format_exc())
                raise HTTPException(
                    status_code=400, 
                    detail=f"模型处理错误: {str(e)}"
                )

            # 创建可视化推理图像
            try:
                reasoning_image, metadata = create_reasoning_image(image, generated_text, self.tags)
                
                # 添加步骤信息到元数据
                if step_num is not None:
                    metadata["step"] = step_num
                
                # 添加faithfulness分数到元数据
                if faith_scores is not None:
                    metadata["faith_scores"] = faith_scores.tolist()
                
                # 保存数据到文件夹
                if self.save_files:
                    save_dir = save_prediction_data(
                        image, 
                        action, 
                        reasoning_image, 
                        generated_text, 
                        metadata,
                        instruction,
                        faith_scores,
                        faith_actions
                    )
                    if step_num is not None:
                        logging.info(f"步骤 {step_num}: 预测数据已保存到: {save_dir}")
                    else:
                        logging.info(f"预测数据已保存到: {save_dir}")
                
            except Exception as e:
                logging.error(f"创建或保存推理数据时出错: {str(e)}")
                logging.error(traceback.format_exc())
                # 如果可视化处理失败，仍然返回动作坐标和faith分数
                result = {
                    "action": action.tolist() if hasattr(action, "tolist") else action
                }
                if faith_scores is not None:
                    result["faith_scores"] = faith_scores.tolist() if hasattr(faith_scores, "tolist") else faith_scores
                
                return JSONResponse(result)
            
            # 返回动作和faith分数
            result = {
                "action": action.tolist() if hasattr(action, "tolist") else action
            }
            
            if faith_scores is not None:
                result["faith_scores"] = faith_scores.tolist() if hasattr(faith_scores, "tolist") else faith_scores
            
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
                "mode": "faith",
                "active_sessions": len(self.sessions)
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
                if session_id in self.session_steps:
                    del self.session_steps[session_id]
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
    """启动ECot Faithfulness评估服务器"""

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 检查predictions目录
    if cfg.save_files:
        check_predictions_directory()
    
    server = ECotServerFaith(
        cfg.model_path, 
        save_files=cfg.save_files,
        history_adaptive=cfg.history_adaptive
    )
    
    logging.info("Faithfulness评估服务器已启动")
    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    run_server()
