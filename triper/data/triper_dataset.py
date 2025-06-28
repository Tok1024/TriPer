import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchaudio
import json
from typing import Optional, Dict, Any, List

class TriperDataset(Dataset):
    """
    Triper数据集类，用于加载包含音频、图像和对话的多模态数据
    """

    def __init__(self, 
                 json_path: str, 
                 media_root_path: str, 
                 image_processor=None, 
                 audio_processor=None,
                 max_audio_length: int = 16000 * 10,  # 10秒音频
                 default_image_size: tuple = (224, 224)):
        """
        Args:
            json_path (str): 指向数据集JSON文件的路径
            media_root_path (str): 存放所有媒体文件的根目录
            image_processor: 用于处理图像的处理器
            audio_processor: 用于处理音频的处理器
            max_audio_length (int): 最大音频长度（样本数）
            default_image_size (tuple): 默认图像尺寸
        """
        super().__init__()
        
        # 加载JSON数据
        print(f"正在从以下路径加载数据集描述文件: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
        print(f"发现 {len(self.data_list)} 个数据样本。")

        # 存储配置
        self.media_root_path = media_root_path
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.max_audio_length = max_audio_length
        self.default_image_size = default_image_size

    def __len__(self) -> int:
        """返回数据集中的样本总数"""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        根据索引获取单个数据样本
        
        Returns:
            Dict包含以下键:
            - id: 样本ID
            - image: 处理后的图像张量或PIL图像
            - audio: 处理后的音频张量
            - conversation: 对话历史列表
            - scene_description: 场景描述
            - metadata: 额外的元数据信息
        """
        sample = self.data_list[idx]
        
        # 1. 加载音频
        audio_tensor = self._load_audio(sample)
        
        # 2. 加载图像
        image = self._load_image(sample)
        
        # 3. 处理对话数据
        conversation = self._process_conversation(sample)
        
        # 4. 获取场景描述
        scene_description = self._get_scene_description(sample)
        
        # 5. 准备元数据
        metadata = self._extract_metadata(sample)
        
        return {
            "id": sample['id'],
            "image": image,
            "audio": audio_tensor,
            "conversation": conversation,
            "scene_description": scene_description,
            "metadata": metadata
        }

    def _load_audio(self, sample: Dict) -> torch.Tensor:
        """加载和处理音频文件"""
        audio_path = os.path.join(self.media_root_path, sample['audio'])
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 确保是单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 限制音频长度
            if waveform.shape[1] > self.max_audio_length:
                waveform = waveform[:, :self.max_audio_length]
            elif waveform.shape[1] < self.max_audio_length:
                # 如果音频太短，用零填充
                padding = self.max_audio_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # 应用音频处理器
            if self.audio_processor:
                waveform = self.audio_processor(waveform, sampling_rate=sample_rate)
                
        except (FileNotFoundError, Exception) as e:
            print(f"警告: 无法加载音频文件 {audio_path}: {e}")
            print("使用静音作为替代")
            waveform = torch.zeros((1, self.max_audio_length))
            
        return waveform

    def _load_image(self, sample: Dict) -> Any:
        """加载和处理图像文件"""
        # 从视频路径构造图像路径
        video_filename = sample['video']
        image_filename = video_filename.replace('.mp4', '.jpg')
        image_path = os.path.join(self.media_root_path, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # 应用图像处理器
            if self.image_processor:
                processed = self.image_processor(image, return_tensors="pt")
                if isinstance(processed, dict) and 'pixel_values' in processed:
                    image = processed['pixel_values'][0]
                else:
                    image = processed
                    
        except (FileNotFoundError, Exception) as e:
            print(f"警告: 无法加载图像文件 {image_path}: {e}")
            print("使用空白图像作为替代")
            image = Image.new('RGB', self.default_image_size, (255, 255, 255))
            
            if self.image_processor:
                processed = self.image_processor(image, return_tensors="pt")
                if isinstance(processed, dict) and 'pixel_values' in processed:
                    image = processed['pixel_values'][0]
                    
        return image

    def _process_conversation(self, sample: Dict) -> List[Dict[str, str]]:
        """处理对话数据"""
        conversation = []
        
        # 确保所有对话相关字段的长度一致
        speakers = sample.get('speaker', [])
        texts = sample.get('text', [])
        emotions = sample.get('emotion', [])
        
        # 取最短长度以防数据不一致
        min_length = min(len(speakers), len(texts), len(emotions))
        
        for i in range(min_length):
            turn = {
                'speaker': speakers[i] if speakers[i] else "Unknown",
                'text': texts[i] if texts[i] else "",
                'emotion': emotions[i] if emotions[i] else "neutral"
            }
            conversation.append(turn)
            
        return conversation

    def _get_scene_description(self, sample: Dict) -> str:
        """获取场景描述"""
        scene_desc = sample.get('scene description', "")
        
        # 如果是列表，取第一个元素
        if isinstance(scene_desc, list) and len(scene_desc) > 0:
            return scene_desc[0]
        elif isinstance(scene_desc, str):
            return scene_desc
        else:
            return ""

    def _extract_metadata(self, sample: Dict) -> Dict[str, Any]:
        """提取额外的元数据信息"""
        metadata = {
            'season': sample.get('season', ''),
            'episode': sample.get('episode', ''),
            'scene': sample.get('scene', ''),
            'utterance': sample.get('utterance', ''),
            'camera': sample.get('camera', []),
            'video_file': sample.get('video', ''),
            'audio_file': sample.get('audio', '')
        }
        return metadata

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """获取样本的基本信息（不加载媒体文件）"""
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_list)}")
            
        sample = self.data_list[idx]
        return {
            'id': sample['id'],
            'speaker_count': len(sample.get('speaker', [])),
            'text_length': sum(len(text) for text in sample.get('text', [])),
            'emotions': sample.get('emotion', []),
            'has_video': bool(sample.get('video')),
            'has_audio': bool(sample.get('audio')),
            'scene_info': f"S{sample.get('season', '?')}E{sample.get('episode', '?')}",
        }

    def validate_dataset(self) -> Dict[str, Any]:
        """验证数据集的完整性"""
        stats = {
            'total_samples': len(self.data_list),
            'missing_audio': 0,
            'missing_video': 0,
            'empty_conversations': 0,
            'emotion_distribution': {},
            'speaker_distribution': {}
        }
        
        for sample in self.data_list:
            # 检查音频文件
            audio_path = os.path.join(self.media_root_path, sample.get('audio', ''))
            if not os.path.exists(audio_path):
                stats['missing_audio'] += 1
                
            # 检查视频/图像文件
            video_file = sample.get('video', '')
            image_file = video_file.replace('.mp4', '.jpg')
            image_path = os.path.join(self.media_root_path, image_file)
            if not os.path.exists(image_path):
                stats['missing_video'] += 1
                
            # 检查对话
            if not sample.get('text') or all(not text for text in sample.get('text', [])):
                stats['empty_conversations'] += 1
                
            # 统计情感分布
            for emotion in sample.get('emotion', []):
                stats['emotion_distribution'][emotion] = stats['emotion_distribution'].get(emotion, 0) + 1
                
            # 统计说话人分布
            for speaker in sample.get('speaker', []):
                if speaker:  # 跳过空说话人
                    stats['speaker_distribution'][speaker] = stats['speaker_distribution'].get(speaker, 0) + 1
        
        return stats