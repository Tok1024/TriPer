import torch
from dataclasses import dataclass
from typing import Dict, Sequence, Optional
import transformers
from PIL import Image

@dataclass
class TriperDataCollator:
    """Triper多模态数据整理器 - 在这里进行实际的数据处理"""
    
    tokenizer: transformers.PreTrainedTokenizer
    image_processor: Optional[object] = None
    audio_processor: Optional[object] = None
    max_length: int = 2048

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """处理一个batch的数据"""
        batch_size = len(instances)
        
        # 1. 处理对话文本
        conversations = [self._build_conversation_text(inst) for inst in instances]
        
        # Tokenize对话
        tokenized = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        batch = {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "labels": tokenized.input_ids.clone(),  # 简单情况下
        }

        # 2. 处理图像数据 - 修正图像尺寸！
        if 'image_path' in instances[0]:
            images = []
            for inst in instances:
                if inst.get('has_image', False):
                    # 动态加载图像
                    from triper.data.triper_dataset import TriperDataset
                    temp_dataset = TriperDataset.__new__(TriperDataset)
                    temp_dataset.default_image_size = (336, 336)  # 修改为模型期望的尺寸
                    image = temp_dataset._load_image_raw(inst['image_path'])
                else:
                    image = Image.new('RGB', (336, 336), (255, 255, 255))  # 修改为336x336
                images.append(image)
            
            # 应用图像处理器
            if self.image_processor:
                try:
                    # 批量处理图像，让处理器自己决定尺寸
                    processed = self.image_processor(images, return_tensors="pt")
                    
                    # 处理不同类型的返回值
                    if isinstance(processed, dict):
                        if 'pixel_values' in processed:
                            batch['images'] = processed['pixel_values']
                            print(f"🖼️ Image batch shape: {batch['images'].shape}")
                        elif 'input_ids' in processed:
                            batch['images'] = processed['input_ids']
                        else:
                            # 如果有其他键，取第一个张量
                            for key, value in processed.items():
                                if isinstance(value, torch.Tensor):
                                    batch['images'] = value
                                    print(f"🖼️ Image batch shape: {batch['images'].shape}")
                                    break
                    else:
                        # 如果直接返回张量
                        batch['images'] = processed
                        print(f"🖼️ Image batch shape: {batch['images'].shape}")
                        
                except Exception as e:
                    print(f"❌ 图像处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 使用默认图像张量，确保使用正确的尺寸
                    import torchvision.transforms as transforms
                    transform = transforms.Compose([
                        transforms.Resize((336, 336)),  # 修改为336x336
                        transforms.ToTensor(),
                    ])
                    image_tensors = [transform(img) for img in images]
                    batch['images'] = torch.stack(image_tensors)
            else:
                # 如果没有处理器，手动转换为张量，使用正确的尺寸
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((336, 336)),  # 修改为336x336
                    transforms.ToTensor(),
                ])
                image_tensors = [transform(img) for img in images]
                batch['images'] = torch.stack(image_tensors)

        # 3. 处理音频数据
        if 'audio_path' in instances[0]:
            audio_features = []
            for inst in instances:
                if inst.get('has_audio', False):
                    audio_path = inst['audio_path']
                    
                    if self.audio_processor:
                        try:
                            # 直接传入音频路径，让音频编码器处理
                            audio_feat = self.audio_processor(audio_path)
                            
                            # 确保返回的是纯张量
                            if hasattr(audio_feat, 'data'):
                                audio_feat = audio_feat.data
                            elif isinstance(audio_feat, dict):
                                # 如果是字典，取第一个张量
                                for key, value in audio_feat.items():
                                    if isinstance(value, torch.Tensor):
                                        audio_feat = value
                                        break
                            
                            # 如果返回的是batch格式，取第一个
                            if audio_feat.dim() == 3 and audio_feat.shape[0] == 1:
                                audio_feat = audio_feat.squeeze(0)  # [seq_len, hidden_dim]
                            
                            print(f"🎵 Audio feature shape: {audio_feat.shape}")
                            
                        except Exception as e:
                            print(f"❌ 音频处理失败: {e}")
                            import traceback
                            traceback.print_exc()
                            # 使用默认维度的零张量作为备用
                            audio_feat = torch.zeros(64, 1280)  # [seq_len, hidden_dim]
                    else:
                        # 如果没有音频处理器，加载原始音频
                        from triper.data.triper_dataset import TriperDataset
                        temp_dataset = TriperDataset.__new__(TriperDataset)
                        temp_dataset.max_audio_length = 16000 * 10
                        waveform = temp_dataset._load_audio_raw(audio_path)
                        audio_feat = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
                else:
                    # 没有音频文件时的默认值
                    audio_feat = torch.zeros(64, 1280)  # [seq_len, hidden_dim]
                    
                audio_features.append(audio_feat)
            
            # Stack所有音频特征
            if audio_features:
                batch['audio_features'] = torch.stack(audio_features)
                print(f"🎵 Final audio batch shape: {batch['audio_features'].shape}")

        return batch

    def _build_conversation_text(self, instance: Dict) -> str:
        """构建用于训练的对话文本"""
        text_parts = []
        
        # 添加场景描述
        scene_desc = instance.get('scene_description', '')
        if scene_desc:
            text_parts.append(f"Scene: {scene_desc}")
        
        # 添加对话内容
        conversation = instance.get('conversation', [])
        for turn in conversation:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            emotion = turn.get('emotion', 'neutral')
            if text:  # 只添加非空对话
                text_parts.append(f"{speaker} ({emotion}): {text}")
        
        # 如果没有任何内容，返回默认文本
        if not text_parts:
            text_parts.append("No conversation available.")
            
        return "\n".join(text_parts)