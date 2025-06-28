import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import MaxNLocator, LogLocator, ScalarFormatter
from scipy.signal import argrelextrema


# ======================
# 关键帧提取工具核心类定义
# ======================
class Frame:
    """存储采样帧信息的类
    Attributes:
        id (int): 采样后的帧ID（非原始视频帧号）
        diff (float): 当前帧与前一帧的平均像素差异值
    """

    def __init__(self, sampled_id, avg_diff):
        self.sampled_id = sampled_id  # 采样后的帧ID（从0开始计数）
        self.avg_diff = avg_diff  # 帧间平均像素差异（用于关键帧判断）

    def __lt__(self, other):
        return self.sampled_id < other.sampled_id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.sampled_id == other.sampled_id


def smooth(data_array, window_size=7, window_type='hanning'):
    """使用滑动窗口对数据进行平滑处理
    Args:
        data_array (np.ndarray): 输入数据数组（一维）
        window_size (int): 滑动窗口大小（需为正整数）
        window_type (str): 窗口类型，支持'flat'（移动平均）或numpy中的窗口函数名（如'hanning'）
    Returns:
        np.ndarray: 平滑后的数组（长度与输入一致）
    """
    print(f"候选关键帧帧数：{len(data_array)}，平滑窗口大小：{window_size}")

    # 扩展数组两端以处理边界效应
    extended_data = np.r_[
        2 * data_array[0] - data_array[window_size:1:-1],
        data_array,
        2 * data_array[-1] - data_array[-1:-window_size:-1]
    ]

    # 生成窗口函数
    if window_type == 'flat':
        window = np.ones(window_size, dtype='d')
    else:
        window = getattr(np, window_type)(window_size)

    # 卷积计算平滑后的数据
    smoothed_data = np.convolve(window / window.sum(), extended_data, mode='same')
    # 去除扩展部分，返回原始长度的平滑数据
    return smoothed_data[window_size - 1:-window_size + 1]


# ======================
# 主处理流程
# ======================
def main():
    t_start = time.time()  # 开始计时（总处理阶段）

    # ----------------------
    # 初始化参数配置
    # ----------------------
    video_path = "./simple_test_vedio.mp4"  # 输入视频路径
    output_base_dir = "./keyframe_output_dir/"  # 关键帧输出基础目录

    # 生成输出文件夹路径（基于视频文件名）
    video_filename = os.path.basename(video_path)  # 提取文件名（包含扩展名）
    video_name = video_filename.replace('.mp4', '')  # 生成无扩展名的文件夹名
    output_dir = output_base_dir + video_name + "/"  # 完整输出路径

    # 创建输出目录（如果不存在）
    if not cv2.os.path.exists(output_dir):
        cv2.os.makedirs(output_dir)
        print(f"创建输出目录：{output_dir}")

    # ----------------------
    # 视频基础信息获取
    # ----------------------
    cap = cv2.VideoCapture(str(video_path))  # 初始化视频读取对象

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    frame_skip = 5  # 帧采样间隔（每隔n帧采样一次）

    print(f"视频总帧数：{total_frame_count}，采样间隔：{frame_skip}帧")

    # ----------------------
    # 帧采样与差异计算
    # ----------------------
    sampled_frame_positions = []  # 存储采样帧的原始位置（如第0帧、第5帧、第10帧...）
    sampled_frames = []  # 存储采样帧的信息（包含ID和差异值）

    prev_frame = None  # 前一帧图像（用于差分计算）
    current_frame_pos = 0  # 当前处理的视频帧位置（原始帧号）
    sampled_id = 0  # 采样后的帧ID（从0开始计数）

    t0_start = time.time()  # 开始计时（差异计算阶段）

    while current_frame_pos < total_frame_count:
        # 设置并读取当前采样帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
        success, current_frame = cap.read()

        if not success:
            print("视频帧读取失败，终止处理")
            break

        # 计算帧间差异（仅当前一帧存在时）
        if prev_frame is not None:
            # 计算绝对差分并求平均像素差异
            frame_diff = cv2.absdiff(current_frame, prev_frame)
            avg_diff = np.sum(frame_diff) / (frame_diff.shape[0] * frame_diff.shape[1])

            # 存储采样帧信息
            sampled_frames.append(Frame(sampled_id, avg_diff))
            sampled_id += 1

        # 更新前一帧和帧位置
        prev_frame = current_frame
        current_frame_pos += frame_skip
        sampled_frame_positions.append(current_frame_pos)  # 记录原始帧位置

    cap.release()  # 释放视频对象
    print(f"采样及差异计算完成，共获取{len(sampled_frames)}个候选帧")
    print(f"耗时：{time.time() - t0_start:.4f}秒")

    # ----------------------
    # 关键帧检测（局部最大值法）
    # ----------------------
    if not sampled_frames:
        print("未获取到有效候选帧，处理终止")
        return

    # 提取差异值数组并备份原始数据
    diff_values = np.array([frame.avg_diff for frame in sampled_frames])
    original_diff = diff_values.copy()

    keyframe_indices = set()  # 存储关键帧的采样ID（sampled_id）

    print("\n--------------- 局部最大值关键帧检测 ---------------")
    candidate_count = len(diff_values)
    window_size = max(candidate_count // 7, 1)  # 动态计算平滑窗口大小

    if window_size > 1:
        # 数据平滑处理
        smoothed_diff = smooth(diff_values, window_size)
        # 寻找局部最大值索引（周围点都更小）
        local_max_indices = argrelextrema(smoothed_diff, np.greater)[0]
    else:
        local_max_indices = np.array([])

    # 处理极端情况：局部最大值不足时使用等间距采样
    if len(local_max_indices) < 5 or not local_max_indices.size:
        print("局部最大值不足，切换为等间距关键帧提取")
        step = max(candidate_count // 5, 1)
        local_max_indices = np.arange(0, candidate_count, step)

    print(f"检测到{len(local_max_indices)}个局部最大值索引：{local_max_indices}")

    # 转换为采样ID并去重
    for idx in local_max_indices:
        if 0 <= idx < candidate_count:
            keyframe_indices.add(idx)

    # ----------------------
    # 保存差异趋势图
    # ----------------------
    plt.figure(figsize=(40, 20))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

    plt.stem(original_diff, label='Original Frame Differences')
    plt.yscale('log', base=10)

    plt.xlabel('Sampled Frame Index', fontsize=20, weight='bold')
    plt.ylabel('Average Pixel Difference', fontsize=20, weight='bold')
    plt.title('Frame Difference Trend', fontsize=24, weight='bold')

    plt.savefig(os.path.join(output_dir, 'difference_trend.png'))
    plt.close()
    print("差异趋势图已保存")

    # ----------------------
    # 关键帧提取与保存
    # ----------------------
    t1_start = time.time()
    cap = cv2.VideoCapture(str(video_path))

    for sampled_id in keyframe_indices:
        # 获取原始视频帧位置
        original_frame_pos = sampled_frame_positions[sampled_id]
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_pos)

        success, keyframe = cap.read()
        if success and keyframe is not None:
            # 生成文件名（注意：sampled_id从0开始，文件序号从1开始）
            file_name = f"keyframe_{sampled_id + 1:03d}.png"
            save_path = os.path.join(output_dir, file_name)

            # 保存图像（OpenCV默认BGR格式，直接保存即可）
            cv2.imwrite(save_path, keyframe)
            print(f"已保存关键帧：{file_name}")
        else:
            print(f"警告：读取第{original_frame_pos}帧失败")

    cap.release()
    print(f"\n处理完成，共提取{len(keyframe_indices)}个关键帧")
    print(f"总耗时：{time.time() - t_start:.4f}秒")
    print("=" * 50)


if __name__ == "__main__":
    main()