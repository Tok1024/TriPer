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
    只返回原始数据路径和元信息，具体处理交给collator
    """

    def __init__(self, 
                 json_path: str, 
                 media_root_path: str, 
                 mode: str = "raw",  # "raw" 或 "processed"
                 max_audio_length: int = 16000 * 10,
                 default_image_size: tuple = (224, 224)):
        """
        Args:
            json_path (str): 指向数据集JSON文件的路径
            media_root_path (str): 存放所有媒体文件的根目录
            mode (str): "raw" 只返回路径，"processed" 返回处理后的数据
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
        self.audio_dir = os.path.join(media_root_path, "audio")
        self.video_dir = os.path.join(media_root_path, "video")
        self.images_dir = os.path.join(media_root_path, "images")
        self.mode = mode
        self.max_audio_length = max_audio_length
        self.default_image_size = default_image_size
        
        print(f"数据集模式: {mode}")
        print(f"音频文件夹: {self.audio_dir}")
        print(f"视频文件夹: {self.video_dir}")
        print(f"图像文件夹: {self.images_dir}")

    def __len__(self) -> int:
        """返回数据集中的样本总数"""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """根据索引获取单个数据样本"""
        sample = self.data_list[idx]
        
        if self.mode == "raw":
            return self._get_raw_sample(sample)
        else:
            return self._get_processed_sample(sample)

    def _get_raw_sample(self, sample: Dict) -> Dict[str, Any]:
        """返回原始数据（路径和元信息）"""
        # 构建文件路径
        audio_filename = sample['audio']
        video_filename = sample['video']
        image_filename = video_filename.replace('.mp4', '.jpg')
        
        audio_path = os.path.join(self.audio_dir, audio_filename)
        image_path = os.path.join(self.images_dir, image_filename)
        
        # 处理对话数据
        conversation = self._process_conversation(sample)
        scene_description = self._get_scene_description(sample)
        
        return {
            "id": sample['id'],
            "audio_path": audio_path,
            "image_path": image_path,
            "conversation": conversation,
            "scene_description": scene_description,
            "metadata": self._extract_metadata(sample),
            # 添加验证标志
            "has_audio": os.path.exists(audio_path),
            "has_image": os.path.exists(image_path),
        }

    def _get_processed_sample(self, sample: Dict) -> Dict[str, Any]:
        """返回处理后的数据（向后兼容）"""
        # 保留原有的处理逻辑，用于向后兼容
        audio_tensor = self._load_audio(sample)
        image = self._load_image(sample)
        conversation = self._process_conversation(sample)
        scene_description = self._get_scene_description(sample)
        
        return {
            "id": sample['id'],
            "image": image,
            "audio": audio_tensor,
            "conversation": conversation,
            "scene_description": scene_description,
            "metadata": self._extract_metadata(sample)
        }

    def _load_audio_raw(self, audio_path: str) -> torch.Tensor:
        """加载原始音频并预提取特征"""
        try:
            if hasattr(self, '_audio_cache'):
                if audio_path in self._audio_cache:
                    return self._audio_cache[audio_path]
            
            # 使用外部音频编码器预提取特征
            if hasattr(self, 'audio_encoder') and self.audio_encoder is not None:
                with torch.no_grad():
                    features = self.audio_encoder.extract_features_from_audio(audio_path)
                    # 缓存特征
                    if not hasattr(self, '_audio_cache'):
                        self._audio_cache = {}
                    self._audio_cache[audio_path] = features
                    return features
            
            # 如果没有编码器，返回原始音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 确保是单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 限制音频长度
            if waveform.shape[1] > self.max_audio_length:
                waveform = waveform[:, :self.max_audio_length]
            elif waveform.shape[1] < self.max_audio_length:
                padding = self.max_audio_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
            return waveform
                
        except Exception as e:
            print(f"警告: 无法加载音频文件 {audio_path}: {e}")
            return torch.zeros(64, 1280)  # 默认特征

    def _load_image_raw(self, image_path: str) -> Image.Image:
        """加载原始图像（用于collator调用）"""
        try:
            if os.path.exists(image_path):
                return Image.open(image_path).convert('RGB')
            else:
                # 返回空白图像
                return Image.new('RGB', self.default_image_size, (255, 255, 255))
        except Exception as e:
            print(f"警告: 无法加载图像文件 {image_path}: {e}")
            return Image.new('RGB', self.default_image_size, (255, 255, 255))

    # 保留原有方法用于向后兼容
    def _load_audio(self, sample: Dict) -> torch.Tensor:
        """加载和处理音频文件（向后兼容）"""
        audio_filename = sample['audio']
        audio_path = os.path.join(self.audio_dir, audio_filename)
        return self._load_audio_raw(audio_path)

    def _load_image(self, sample: Dict) -> Image.Image:
        """加载和处理图像文件（向后兼容）"""
        video_filename = sample['video']
        image_filename = video_filename.replace('.mp4', '.jpg')
        image_path = os.path.join(self.images_dir, image_filename)
        return self._load_image_raw(image_path)

    def _process_conversation(self, sample: Dict) -> List[Dict[str, str]]:
        """处理对话数据"""
        conversation = []
        
        speakers = sample.get('speaker', [])
        texts = sample.get('text', [])
        emotions = sample.get('emotion', [])
        
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