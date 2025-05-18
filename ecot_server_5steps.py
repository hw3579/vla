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


# 检查predictions目录
def check_predictions_directory(directory="predictions", warn_if_non_empty=True):
    """
    检查predictions目录是否存在且是否非空
    
    Args:
        directory: 要检查的目录名称
        warn_if_non_empty: 如果目录非空是否显示警告
        
    Returns:
        bool: 目录是否存在且非空
    """
    # 确保目录存在
    if not os.path.exists(directory):
        os.makedirs(directory)
        return False
    
    # 检查目录是否为空
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




# === 实用工具函数 ===
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
    #####
        # 使用 truetype 加载字体并指定大小
    try:
        # 尝试加载一个系统字体
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except IOError:
        try:
            # 如果找不到，尝试另一个常见字体
            font = ImageFont.truetype("Arial.ttf", 14)
        except IOError:
            # 如果都找不到，使用默认字体
            font = ImageFont.load_default()
            logging.warning("无法加载指定字体，使用默认字体")
    #####
    #     解释
    # 问题出现的原因是:

    # 在某些旧版本的 PIL/Pillow 中，ImageFont.load_default() 不接受任何参数
    # 在较新版本中，它可能支持 size 参数
    # 最安全的方法是使用 truetype() 方法来加载指定大小的字体，如方法二所示。这样无论您的 Pillow 版本如何，代码都能正常工作。
    ####
    # font = ImageFont.load_default(size=14)
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


def save_prediction_data(image, action, reasoning_image, generated_text, metadata, instruction):
    """保存预测数据到文件夹"""
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
    
    # 保存元数据和动作
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        # 确保numpy数组可以序列化
        action_list = action.tolist() if hasattr(action, "tolist") else action
        json.dump({
            "action": action_list,
            "metadata": metadata,
            "timestamp": timestamp,
            "instruction": instruction
        }, f, indent=2)
    
    return save_dir









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
                        instruction
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
    use_5step: bool = False                        # 是否使用5step模式



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