import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchaudio

class TriperDataset(Dataset):

    def __init__(self, json_path, media_root_path, image_processor=None, audio_processor=None):
        """
        Args:
            json_path (str): 指向数据集JSON文件的路径。
            media_root_path (str): 存放所有媒体文件（视频、音频）的根目录。
            image_processor: 用于处理图像的处理器 。
            audio_processor: 用于处理音频的处理器 。
        """
        super().__init__()
        # 1. 加载JSON描述文件
        print(f"正在从以下路径加载数据集描述文件: {json_path}")
        self.data_list = pd.read_json(json_path, orient='records')
        print(f"发现 {len(self.data_list)} 个数据样本。")

        # 2. 存储路径和处理器
        self.media_root_path = media_root_path
        self.image_processor = image_processor
        self.audio_processor = audio_processor

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        根据索引 (idx) 获取单个数据样本。
        """
        # 1. 根据索引获取该样本的元数据字典
        sample_meta = self.data_list[idx]

        # --- 2. 加载媒体文件 ---
        
        # 加载音频
        # 构建音频文件的完整路径
        audio_path = os.path.join(self.media_root_path, sample_meta['audio'])
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            # 音频处理器
            if self.audio_processor:
                waveform = self.audio_processor(waveform)
        except FileNotFoundError:
            print(f"警告: 找不到音频文件 {audio_path}，将使用静音作为替代。")
            waveform = torch.zeros((1, 16000)) # 1秒的静音

        # 加载图像
        # **注意**: JSON里是 "video" 路径, 但一期模型需要的是图像。
        # 需要一个预处理步骤，从视频中提取关键帧并保存为图片。
        # 这里假设已经提取好，并且图片和视频文件名相关。
        image_path_placeholder = os.path.join(self.media_root_path, sample_meta['video'].replace('.mp4', '.jpg'))
        try:
            image = Image.open(image_path_placeholder).convert('RGB')
            if self.image_processor:
                image = self.image_processor(image, return_tensors="pt")['pixel_values'][0]
        except FileNotFoundError:
            print(f"警告: 找不到图片文件 {image_path_placeholder}，将使用空白图片作为替代。")
            image = Image.new('RGB', (224, 224), (255, 255, 255))

        # 3. 构造对话文本
        # 将多轮对话拼接成一个完整的上下文
        conversation_history = []
        for i in range(len(sample_meta['speaker'])):
            turn = {
                'speaker': sample_meta['speaker'][i],
                'text': sample_meta['text'][i],
                'emotion': sample_meta['emotion'][i]
            }
            conversation_history.append(turn)

        # 4. 将所有处理好的数据打包成一个字典返回
        return {
            "id": sample_meta['id'],
            "image": image,
            "audio": waveform,
            "conversation": conversation_history,
            "scene_description": sample_meta['scene description']
        }