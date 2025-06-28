"""
批量提取视频关键帧脚本
对data/video文件夹下的所有mp4视频文件提取一个关键帧
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from scipy.signal import argrelextrema


class Frame:
    """存储采样帧信息的类"""
    def __init__(self, sampled_id, avg_diff):
        self.sampled_id = sampled_id
        self.avg_diff = avg_diff


def smooth(data_array, window_size=7):
    """数据平滑处理"""
    if len(data_array) < window_size:
        return data_array
    
    extended_data = np.r_[
        2 * data_array[0] - data_array[window_size:1:-1],
        data_array,
        2 * data_array[-1] - data_array[-1:-window_size:-1]
    ]
    
    window = np.hanning(window_size)
    smoothed_data = np.convolve(window / window.sum(), extended_data, mode='same')
    return smoothed_data[window_size - 1:-window_size + 1]


def extract_single_keyframe(video_path, output_path):
    """
    从视频中提取一个关键帧
    
    Args:
        video_path (str): 视频文件路径
        output_path (str): 输出图像路径
    
    Returns:
        bool: 是否成功提取关键帧
    """
    print(f"处理视频: {os.path.basename(video_path)}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ 无法打开视频文件")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        print(f"  ✗ 视频帧数不足")
        cap.release()
        return False
    
    # 采样参数
    frame_skip = max(1, total_frames // 50)  # 最多采样50帧
    sampled_frames = []
    prev_frame = None
    current_pos = 0
    sampled_id = 0
    
    print(f"  总帧数: {total_frames}, 采样间隔: {frame_skip}")
    
    # 采样并计算帧间差异
    while current_pos < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        success, current_frame = cap.read()
        
        if not success:
            break
        
        if prev_frame is not None:
            # 转换为灰度图以加快计算
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # 计算帧间差异
            frame_diff = cv2.absdiff(gray_current, gray_prev)
            avg_diff = np.mean(frame_diff)
            
            sampled_frames.append(Frame(sampled_id, avg_diff))
            sampled_id += 1
        
        prev_frame = current_frame
        current_pos += frame_skip
    
    if len(sampled_frames) < 3:
        print(f"  ✗ 采样帧数不足，直接提取中间帧")
        # 提取中间帧作为关键帧
        middle_frame_pos = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_pos)
        success, keyframe = cap.read()
        
        if success:
            cv2.imwrite(output_path, keyframe)
            print(f"  ✓ 已保存中间帧: {os.path.basename(output_path)}")
            cap.release()
            return True
        else:
            cap.release()
            return False
    
    # 寻找最佳关键帧
    diff_values = np.array([frame.avg_diff for frame in sampled_frames])
    
    # 数据平滑
    window_size = max(3, len(diff_values) // 10)
    if window_size < len(diff_values):
        smoothed_diff = smooth(diff_values, window_size)
    else:
        smoothed_diff = diff_values
    
    # 寻找局部最大值
    local_max_indices = argrelextrema(smoothed_diff, np.greater)[0]
    
    if len(local_max_indices) == 0:
        # 如果没有局部最大值，选择全局最大值
        best_frame_idx = np.argmax(smoothed_diff)
    else:
        # 选择差异值最大的局部最大值
        max_diff_idx = 0
        max_diff_value = 0
        for idx in local_max_indices:
            if smoothed_diff[idx] > max_diff_value:
                max_diff_value = smoothed_diff[idx]
                max_diff_idx = idx
        best_frame_idx = max_diff_idx
    
    # 计算对应的原始帧位置
    original_frame_pos = best_frame_idx * frame_skip + frame_skip
    
    # 提取关键帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_pos)
    success, keyframe = cap.read()
    
    if success:
        cv2.imwrite(output_path, keyframe)
        print(f"  ✓ 已保存关键帧: {os.path.basename(output_path)} (原始帧位置: {original_frame_pos})")
        cap.release()
        return True
    else:
        print(f"  ✗ 提取关键帧失败")
        cap.release()
        return False


def batch_extract_keyframes():
    """批量提取关键帧"""
    # 定义路径
    data_root = "/home/wly/szl_all_code/triper-project/data"
    video_dir = os.path.join(data_root, "video")
    images_dir = os.path.join(data_root, "images")
    json_file = os.path.join(data_root, "simple_data_20_samples.json")
    
    # 创建输出目录
    os.makedirs(images_dir, exist_ok=True)
    print(f"创建图像输出目录: {images_dir}")
    
    # 检查视频目录
    if not os.path.exists(video_dir):
        print(f"错误: 视频目录不存在 - {video_dir}")
        return
    
    # 读取JSON获取视频列表
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        video_list = [sample['video'] for sample in data_list if sample.get('video')]
        print(f"从JSON文件读取到 {len(video_list)} 个视频文件")
    else:
        # 如果没有JSON文件，处理目录下所有mp4文件
        video_list = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
        print(f"在视频目录找到 {len(video_list)} 个mp4文件")
    
    # 统计变量
    success_count = 0
    failed_count = 0
    
    print(f"\n开始批量提取关键帧:")
    print("-" * 60)
    
    for i, video_filename in enumerate(video_list, 1):
        video_path = os.path.join(video_dir, video_filename)
        
        # 生成对应的图像文件名
        image_filename = video_filename.replace('.mp4', '.jpg')
        image_path = os.path.join(images_dir, image_filename)
        
        print(f"[{i:2d}/{len(video_list)}] ", end="")
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"{video_filename} - ✗ 视频文件不存在")
            failed_count += 1
            continue
        
        # 检查图像是否已存在
        if os.path.exists(image_path):
            print(f"{video_filename} - ✓ 关键帧已存在，跳过")
            success_count += 1
            continue
        
        # 提取关键帧
        if extract_single_keyframe(video_path, image_path):
            success_count += 1
        else:
            failed_count += 1
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("批量提取完成！统计结果:")
    print(f"  成功提取: {success_count} 个")
    print(f"  提取失败: {failed_count} 个")
    print(f"  总计处理: {len(video_list)} 个视频文件")
    
    # 验证输出目录
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
        print(f"  输出目录包含: {len(image_files)} 个图像文件")


if __name__ == "__main__":
    print("批量关键帧提取脚本")
    print("=" * 60)
    
    try:
        batch_extract_keyframes()
    except Exception as e:
        print(f"发生错误: {e}")
    
    print("\n脚本执行完成！")