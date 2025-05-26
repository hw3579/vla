"""
该脚本用于将 predictions 文件夹中的所有推理图像整合成一个视频。

使用方法:

python create_reasoning_video.py --predictions_dir ./predictions --output_video reasoning_video.mp4 --fps 5

依赖:
    => `pip install opencv-python numpy`
"""

import os
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
import json
from typing import List, Tuple, Optional
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_timestamp_from_dir(dirname: str) -> Tuple[str, Optional[str]]:
    """从目录名中解析时间戳"""
    match = re.match(r'(\d{8}_\d{6})_([a-f0-9]+)', os.path.basename(dirname))
    if match:
        return match.group(1), match.group(2)
    return os.path.basename(dirname), None

def get_prediction_metadata(prediction_dir: str) -> dict:
    """获取预测目录中的元数据"""
    metadata_path = os.path.join(prediction_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def get_instruction_from_metadata(metadata: dict) -> str:
    """从元数据中获取指令"""
    return metadata.get("instruction", "no instruction")

def get_inference_time_from_metadata(metadata: dict) -> str:
    """从元数据中获取推理时间"""
    inference_time = metadata.get("inference_time", None)
    if inference_time is not None:
        return f"{inference_time:.4f}s"
    return "unknown inference time"


def create_video_from_predictions(
    predictions_dir: str, 
    output_video: str, 
    fps: int = 5,
    width: int = 1280,
    height: int = 480,
    sort_by_time: bool = True
) -> None:
    """
    从predictions目录中的所有推理图像创建视频
    
    Args:
        predictions_dir: 包含预测结果的目录
        output_video: 输出视频的路径
        fps: 视频帧率
        width: 视频宽度
        height: 视频高度
        sort_by_time: 是否按时间戳排序
    """
    # 获取所有prediction子目录
    prediction_dirs = [d for d in glob.glob(os.path.join(predictions_dir, "*")) if os.path.isdir(d)]
    
    if not prediction_dirs:
        logger.error(f"在 {predictions_dir} 中没有找到任何预测目录")
        return
    
    logger.info(f"找到 {len(prediction_dirs)} 个预测目录")
    
    # 按时间戳排序（如果需要）
    if sort_by_time:
        prediction_dirs.sort(key=lambda x: parse_timestamp_from_dir(x)[0])
        
    # 初始化视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for idx, pred_dir in enumerate(prediction_dirs):
        image_path = os.path.join(pred_dir, "reasoning_image.png")
        
        # 跳过没有reasoning_image.png的目录
        if not os.path.exists(image_path):
            logger.warning(f"在 {pred_dir} 中没有找到推理图像，跳过")
            continue
        
        # 读取图像
        image = cv2.imread(image_path)
        
        # 如果图像尺寸不符合要求，进行调整
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height))
        
        # 获取指令（可选）和推理时间
        metadata = get_prediction_metadata(pred_dir)
        instruction = get_instruction_from_metadata(metadata)
        inference_time = get_inference_time_from_metadata(metadata)

        # 添加额外信息到图像
        timestamp, session_id = parse_timestamp_from_dir(pred_dir)
        info_text = f"#{idx+1} | {timestamp} | Inference time: {inference_time} | {instruction}"
        
        # 在图像顶部添加信息条
        info_bar = np.ones((40, width, 3), dtype=np.uint8) * 255
        cv2.putText(info_bar, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # 将信息条与图像合并
        final_image = np.vstack([info_bar, image[:height-40, :, :]])
        
        # 写入视频
        video_writer.write(final_image)
        
        logger.info(f"处理 {idx+1}/{len(prediction_dirs)}: {pred_dir}")
    
    # 释放资源
    video_writer.release()
    logger.info(f"视频已创建: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="将预测目录中的推理图像生成视频")
    parser.add_argument("--predictions_dir", type=str, default="./predictions", 
                        help="包含预测结果的目录路径")
    parser.add_argument("--output_video", type=str, default="reasoning_video.mp4", 
                        help="输出视频的文件名")
    parser.add_argument("--fps", type=int, default=5, 
                        help="视频帧率")
    parser.add_argument("--width", type=int, default=1280, 
                        help="视频宽度")
    parser.add_argument("--height", type=int, default=480, 
                        help="视频高度")
    parser.add_argument("--no_sort", action="store_false", dest="sort_by_time",
                        help="不按时间戳排序图像")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    create_video_from_predictions(
        args.predictions_dir,
        args.output_video,
        args.fps,
        args.width,
        args.height,
        args.sort_by_time
    )

if __name__ == "__main__":
    main()