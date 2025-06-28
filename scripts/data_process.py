#!/usr/bin/env python3
"""
数据集文件整理脚本
从每个样本目录中读取音频和视频文件，重命名并整理到对应文件夹
"""

import os
import shutil
import json
from pathlib import Path

def organize_dataset():
    """整理数据集文件"""
    
    # 定义路径
    data_root = "/home/wly/szl_all_code/triper-project/data"
    small_dir = os.path.join(data_root, "small")
    json_file = os.path.join(data_root, "simple_data_20_samples.json")
    
    # 创建目标文件夹
    audio_dir = os.path.join(data_root, "audio")
    video_dir = os.path.join(data_root, "video")
    
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"创建目录:")
    print(f"  音频目录: {audio_dir}")
    print(f"  视频目录: {video_dir}")
    
    # 读取JSON文件获取文件名映射
    print(f"\n读取JSON文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # 统计变量
    copied_audio = 0
    copied_video = 0
    missing_folders = 0
    
    print(f"\n开始处理 {len(data_list)} 个样本:")
    print("-" * 60)
    
    for i, sample in enumerate(data_list, 1):
        sample_id = sample['id']
        target_audio_name = sample['audio']  # 目标音频文件名: 00001.wav
        target_video_name = sample['video']  # 目标视频文件名: 00001.mp4
        
        print(f"处理样本 {i:2d}: ID={sample_id}")
        
        # 构造样本文件夹路径（使用音频文件名前缀作为文件夹名）
        folder_name = target_audio_name.split('.')[0]  # 00001
        sample_folder = os.path.join(small_dir, folder_name)
        
        if not os.path.exists(sample_folder):
            print(f"  ✗ 文件夹不存在: {sample_folder}")
            missing_folders += 1
            continue
        
        # 获取文件夹中的所有文件
        files_in_folder = os.listdir(sample_folder)
        
        # 查找音频文件（任何音频格式）
        audio_files = [f for f in files_in_folder if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac'))]
        if audio_files:
            audio_src = os.path.join(sample_folder, audio_files[0])
            audio_dst = os.path.join(audio_dir, target_audio_name)
            
            try:
                shutil.copy2(audio_src, audio_dst)
                copied_audio += 1
                print(f"  ✓ 音频: {audio_files[0]} -> {target_audio_name}")
            except Exception as e:
                print(f"  ✗ 音频复制失败: {e}")
        else:
            print(f"  ✗ 未找到音频文件")
        
        # 查找视频文件（任何视频格式）
        video_files = [f for f in files_in_folder if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv'))]
        if video_files:
            video_src = os.path.join(sample_folder, video_files[0])
            video_dst = os.path.join(video_dir, target_video_name)
            
            try:
                shutil.copy2(video_src, video_dst)
                copied_video += 1
                print(f"  ✓ 视频: {video_files[0]} -> {target_video_name}")
            except Exception as e:
                print(f"  ✗ 视频复制失败: {e}")
        else:
            print(f"  ✗ 未找到视频文件")
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("整理完成！统计结果:")
    print(f"  成功复制音频文件: {copied_audio} 个")
    print(f"  成功复制视频文件: {copied_video} 个")
    print(f"  缺失文件夹: {missing_folders} 个")
    print(f"  总处理样本: {len(data_list)} 个")
    
    # 验证目标文件夹内容
    audio_files = os.listdir(audio_dir) if os.path.exists(audio_dir) else []
    video_files = os.listdir(video_dir) if os.path.exists(video_dir) else []
    
    print(f"\n目标文件夹验证:")
    print(f"  音频文件夹包含: {len(audio_files)} 个文件")
    print(f"  视频文件夹包含: {len(video_files)} 个文件")
    
    return {
        'copied_audio': copied_audio,
        'copied_video': copied_video,
        'missing_folders': missing_folders,
        'total_samples': len(data_list)
    }

def list_sample_folders():
    """列出前几个样本文件夹的内容"""
    data_root = "/home/wly/szl_all_code/triper-project/data"
    small_dir = os.path.join(data_root, "small")
    
    print("样本文件夹内容预览:")
    print("-" * 40)
    
    # 查看前5个文件夹
    for i in range(1, 6):
        folder_name = f"{i:05d}"  # 00001, 00002, etc.
        folder_path = os.path.join(small_dir, folder_name)
        
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            print(f"文件夹 {folder_name}: {files}")
        else:
            print(f"文件夹 {folder_name}: 不存在")

def verify_files():
    """验证文件是否正确复制"""
    data_root = "/home/wly/szl_all_code/triper-project/data"
    json_file = os.path.join(data_root, "simple_data_20_samples.json")
    audio_dir = os.path.join(data_root, "audio")
    video_dir = os.path.join(data_root, "video")
    
    if not os.path.exists(json_file):
        print(f"JSON文件不存在: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print("文件验证结果:")
    print("-" * 40)
    
    for sample in data_list:
        sample_id = sample['id']
        audio_filename = sample['audio']
        video_filename = sample['video']
        
        audio_path = os.path.join(audio_dir, audio_filename)
        video_path = os.path.join(video_dir, video_filename)
        
        audio_exists = "✓" if os.path.exists(audio_path) else "✗"
        video_exists = "✓" if os.path.exists(video_path) else "✗"
        
        print(f"样本 {sample_id}: 音频{audio_exists} 视频{video_exists}")

if __name__ == "__main__":
    print("Triper数据集文件整理脚本")
    print("=" * 60)
    
    # 预览文件夹内容
    list_sample_folders()
    
    print("\n" + "=" * 60)
    
    # 执行整理
    try:
        stats = organize_dataset()
        
        # 验证结果
        print("\n" + "=" * 60)
        verify_files()
        
    except FileNotFoundError as e:
        print(f"错误：找不到文件或目录 - {e}")
    except PermissionError as e:
        print(f"错误：权限不足 - {e}")
    except Exception as e:
        print(f"未预期的错误：{e}")
        
    print("\n脚本执行完成！")