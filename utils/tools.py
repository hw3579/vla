import cv2
import numpy as np
import textwrap
import logging
from PIL import Image, ImageDraw, ImageFont
import enum
import datetime
import uuid
import os
import sys
import json
import json_numpy
json_numpy.patch()



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

# def resize_pos(pos, img_size):
#     return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_gripper(img, pos_list, img_size=(256, 256)):
    for i, pos in enumerate(reversed(pos_list)):
        # 直接使用原始坐标
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

def draw_bboxes(img, bboxes, img_size=(256, 256)):
    for name, bbox in bboxes.items():
        show_name = name
        # 直接使用原始坐标
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            (bbox[0], bbox[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,                # 字体更小
            (0, 0, 0),          # 黑色
            1,
            cv2.LINE_AA,
        )

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


    img_arr = np.array(image)
    draw_gripper(img_arr, metadata["gripper"])
    draw_bboxes(img_arr, bboxes)

    ## 到这里图像的识别和爪子位置没问题

    #分辨率乘2 变为512x512
    img_arr = cv2.resize(img_arr, (512, 512), interpolation=cv2.INTER_LINEAR)


    base = Image.fromarray(np.ones((512, 1024, 3), dtype=np.uint8) * 255)
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

    color = (0,0,0)
    draw.text((0, 0), caption, color, font=font)        

    text_arr = np.array(base)

    if img_arr.shape[0] != text_arr.shape[0]:
        img_arr = cv2.resize(img_arr, (text_arr.shape[1], text_arr.shape[0]))

    reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))
    return reasoning_img, metadata



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


def save_prediction_data(image, action, reasoning_image, generated_text, metadata, instruction, inference_time=None):
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
    save_data = {
        "action": action.tolist() if hasattr(action, "tolist") else action,
        "metadata": metadata,
        "timestamp": timestamp,
        "instruction": instruction
    }
    
    # 添加推理时间到元数据
    if inference_time is not None:
        save_data["inference_time"] = inference_time
    
    with open(metadata_path, "w") as f:
        json.dump(save_data, f, indent=2)
    
    return save_dir