import requests
import json_numpy
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import argparse
import os
import time
import random
import io

# 确保json_numpy可以处理numpy数组的序列化
json_numpy.patch()

def generate_random_image(width=640, height=480):
    """生成一个随机测试图像，包含一些简单物体"""
    # 创建纯白背景
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 物体列表 - 颜色和名称
    objects = [
        ((0, 0, 255), "蓝色方块"),
        ((0, 255, 0), "绿色圆形"),
        ((255, 0, 0), "红色三角形"),
        ((255, 255, 0), "黄色椭圆"),
        ((128, 0, 128), "紫色长方形"),
        ((0, 128, 128), "青色圆环")
    ]
    
    # 从PIL转换为OpenCV便于绘制
    img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 随机位置放置3-6个物体
    placed_objects = []
    num_objects = random.randint(3, 6)
    
    for i in range(num_objects):
        # 随机选择物体
        color, name = random.choice(objects)
        
        # 随机位置
        x = random.randint(50, width - 100)
        y = random.randint(50, height - 100)
        
        # 随机大小
        size = random.randint(30, 80)
        
        # 根据物体类型绘制不同形状
        if "方块" in name or "长方形" in name:
            # 绘制矩形
            w = size
            h = size if "方块" in name else size // 2
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 0, 0), 2)
            placed_objects.append((name, (x, y, x + w, y + h)))
        
        elif "圆形" in name:
            # 绘制圆形
            cv2.circle(img_cv, (x, y), size // 2, color, -1)
            cv2.circle(img_cv, (x, y), size // 2, (0, 0, 0), 2)
            r = size // 2
            placed_objects.append((name, (x - r, y - r, x + r, y + r)))
        
        elif "三角形" in name:
            # 绘制三角形
            pts = np.array([[x, y - size//2], [x - size//2, y + size//2], [x + size//2, y + size//2]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img_cv, [pts], color)
            cv2.polylines(img_cv, [pts], True, (0, 0, 0), 2)
            placed_objects.append((name, (x - size//2, y - size//2, x + size//2, y + size//2)))
        
        elif "椭圆" in name:
            # 绘制椭圆
            cv2.ellipse(img_cv, (x, y), (size//2, size//3), 0, 0, 360, color, -1)
            cv2.ellipse(img_cv, (x, y), (size//2, size//3), 0, 0, 360, (0, 0, 0), 2)
            placed_objects.append((name, (x - size//2, y - size//3, x + size//2, y + size//3)))
        
        elif "圆环" in name:
            # 绘制圆环
            cv2.circle(img_cv, (x, y), size // 2, color, thickness=size // 6)
            cv2.circle(img_cv, (x, y), size // 2, (0, 0, 0), 2)
            r = size // 2
            placed_objects.append((name, (x - r, y - r, x + r, y + r)))
            
    # 添加表格
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 在右下角添加一个简单表格，包含物体名称和位置
    table_x = width - 220
    table_y = height - 30 - len(placed_objects) * 20
    
    draw.rectangle((table_x-5, table_y-5, table_x+215, table_y+5+len(placed_objects)*20), 
                  fill=(240, 240, 240), outline=(0, 0, 0))
    
    draw.text((table_x, table_y-20), "场景中的物体:", fill=(0, 0, 0))
    
    for i, (obj_name, coords) in enumerate(placed_objects):
        y_pos = table_y + i * 20
        draw.text((table_x, y_pos), f"{obj_name}: {coords}", fill=(0, 0, 0))
    
    # 转回numpy数组
    image = np.array(img_pil)
    
    return image, placed_objects

def generate_random_instruction(objects):
    """基于场景中的物体生成随机指令"""
    if not objects:
        return "观察并描述场景"
    
    # 指令模板
    instruction_templates = [
        "将{obj1}放在{obj2}上",
        "拿起{obj1}",
        "移动{obj1}到{obj2}旁边",
        "把{obj1}和{obj2}摆在一起",
        "将{obj1}放在场景中央",
        "帮我整理场景中的物体",
        "把所有{color}的物体组合在一起",
        "找出{obj1}并描述它的位置",
    ]
    
    # 随机选择一个或两个物体
    if len(objects) >= 2:
        obj1, obj2 = random.sample([obj[0] for obj in objects], 2)
        template = random.choice(instruction_templates)
        
        # 处理特殊模板
        if "{color}" in template:
            colors = ["红色", "蓝色", "绿色", "黄色", "紫色", "青色"]
            available_colors = [color for color in colors if any(color in obj[0] for obj in objects)]
            if available_colors:
                color = random.choice(available_colors)
                return template.format(color=color)
            else:
                # 如果没有匹配的颜色，使用另一个模板
                return f"将{obj1}放在{obj2}旁边"
        
        return template.format(obj1=obj1, obj2=obj2)
    else:
        obj1 = objects[0][0]
        # 使用只需要一个物体的模板
        single_obj_templates = [
            "拿起{obj1}",
            "将{obj1}放在场景中央",
            "找出{obj1}并描述它的位置",
        ]
        template = random.choice(single_obj_templates)
        return template.format(obj1=obj1)

def test_ecot_server(image_path=None, instruction=None, host="0.0.0.0", port=8000, session_id=None, save_result=True):
    """
    测试ECot服务器的推理功能
    
    Args:
        image_path: 测试图像的路径
        instruction: 要发送给机器人的指令
        host: 服务器主机地址
        port: 服务器端口
        session_id: 会话ID，用于5step模式
        save_result: 是否保存结果图像
    """
    # 如果未提供图像，生成随机测试图像
    if image_path is None:
        print("未提供图像，生成随机测试图像...")
        image, objects = generate_random_image()
        
        # 如果未提供指令，生成随机指令
        if instruction is None:
            instruction = generate_random_instruction(objects)
            print(f"随机生成指令: {instruction}")
    else:
        # 读取指定图像
        try:
            image = np.array(Image.open(image_path))
            print(f"成功读取图像，形状: {image.shape}")
            
            # 如果未提供指令，生成随机指令
            if instruction is None:
                # 假设图像中有一些物体
                generic_instructions = [
                    "分析场景并执行适当的任务",
                    "找出场景中最明显的物体并拿起它",
                    "整理场景中的物体",
                    "分析图像并找出最适合的操作",
                    "执行最合理的动作"
                ]
                instruction = random.choice(generic_instructions)
                print(f"随机生成指令: {instruction}")
                
        except Exception as e:
            print(f"读取图像时出错: {str(e)}")
            return
    
    # 准备请求数据
    payload = {
        "image": image,
        "instruction": instruction
    }
    
    # 如果使用5step模式，添加会话ID
    if session_id:
        payload["session_id"] = session_id
    elif random.random() < 0.5:  # 50%的概率生成会话ID
        session_id = f"test_session_{int(time.time())}"
        payload["session_id"] = session_id
        print(f"随机生成会话ID: {session_id}")
    
    # 发送请求并计时
    start_time = time.time()
    print(f"向 http://{host}:{port}/act 发送请求...")
    
    try:
        response = requests.post(
            f"http://{host}:{port}/act",
            json=payload,
            timeout=120  # 设置超时时间为2分钟
        )
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        end_time = time.time()
        print(f"请求完成，耗时: {end_time - start_time:.2f}秒")
        
        # 提取结果
        action = result.get("action")
        generated_text = result.get("generated_text")
        
        # 显示推理时间和结果
        print("\n=== 服务器响应 ===")
        print(f"推理时间: {end_time - start_time:.2f}秒")
        print(f"预测动作: {action}")
        
        # 判断是否有reasoning_image
        if "reasoning_image" in result:
            reasoning_image = np.array(result["reasoning_image"])
            
            # 显示推理图像
            plt.figure(figsize=(15, 10))
            plt.imshow(reasoning_image)
            plt.axis('off')
            plt.title(f"指令: {instruction}")
            
            # 保存结果
            if save_result:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_dir = "test_results"
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存原始图像
                input_img_path = f"{save_dir}/input_{timestamp}.png"
                Image.fromarray(image).save(input_img_path)
                print(f"输入图像已保存到: {input_img_path}")
                
                # 保存推理图像
                output_img_path = f"{save_dir}/result_{timestamp}.png"
                plt.savefig(output_img_path)
                print(f"结果图像已保存到: {output_img_path}")
                
                # 保存文本
                text_path = f"{save_dir}/text_{timestamp}.txt"
                with open(text_path, "w") as f:
                    f.write(f"指令: {instruction}\n\n")
                    f.write(f"预测动作: {action}\n\n")
                    if generated_text:
                        f.write("生成文本:\n")
                        f.write(generated_text)
                print(f"结果文本已保存到: {text_path}")
                
                # 如果有会话信息，保存会话状态
                if session_id:
                    step_counter = result.get("step_counter", 0)
                    high_level_frequency = result.get("high_level_frequency", 0)
                    if step_counter or high_level_frequency:
                        session_info = {
                            "session_id": session_id,
                            "step_counter": step_counter,
                            "high_level_frequency": high_level_frequency
                        }
                        session_path = f"{save_dir}/session_{timestamp}.json"
                        with open(session_path, "w") as f:
                            json.dump(session_info, f, indent=2)
                        print(f"会话信息已保存到: {session_path}")
            
            plt.show()
        else:
            print("警告: 响应中没有reasoning_image")
            
            # 仍然保存输入图像和结果
            if save_result:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_dir = "test_results"
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存原始图像
                input_img_path = f"{save_dir}/input_{timestamp}.png"
                Image.fromarray(image).save(input_img_path)
                
                # 保存文本
                text_path = f"{save_dir}/text_{timestamp}.txt"
                with open(text_path, "w") as f:
                    f.write(f"指令: {instruction}\n\n")
                    f.write(f"预测动作: {action}\n\n")
                    if generated_text:
                        f.write("生成文本:\n")
                        f.write(generated_text)
                
        # 打印生成的文本
        if generated_text:
            print("\n=== 生成文本摘要 ===")
            # 只显示文本的前300个字符
            print(f"{generated_text[:300]}...(省略)")
            print(f"完整文本已保存到结果文件中")
            
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试ECot服务器")
    parser.add_argument("--image", type=str, help="测试图像的路径，未指定则随机生成")
    parser.add_argument("--instruction", type=str, help="要发送给机器人的指令，未指定则随机生成")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--session_id", type=str, help="会话ID，用于5step模式")
    parser.add_argument("--no-save", action="store_true", help="不保存结果")
    
    args = parser.parse_args()
    
    test_ecot_server(
        args.image, 
        args.instruction, 
        args.host, 
        args.port, 
        args.session_id,
        not args.no_save
    )