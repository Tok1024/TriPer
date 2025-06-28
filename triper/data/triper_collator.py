import torch
from dataclasses import dataclass
from typing import Dict, Sequence, Optional
import transformers
from PIL import Image
from triper.constants import DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token_batch, process_images
from llava.constants import IMAGE_TOKEN_INDEX

@dataclass
class TriperDataCollator:
    
    tokenizer: transformers.PreTrainedTokenizer
    image_processor: Optional[object] = None
    audio_processor: Optional[object] = None
    model_cfg: Optional[object] = None  # 🔧 新增：接收模型配置
    max_length: int = 2048

    def _build_conversation_text(self, instance: Dict) -> str:
        """构建包含图像token的对话文本"""
        text_parts = []
        
        # 🔧 关键修复：如果有图像，在开头插入图像token
        if instance.get('has_image', False):
            text_parts.append(DEFAULT_IMAGE_TOKEN)
        
        conversation = instance.get('conversation', [])
        for turn in conversation:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            emotion = turn.get('emotion', 'neutral')
            if text:
                text_parts.append(f"{speaker} ({emotion}): {text}")
        
        if len(text_parts) <= 1:  # 只有图像token或为空
            text_parts.append("No conversation available.")
            
        result = "\n".join(text_parts)
        print(f"📝 构建的对话文本（包含图像token）: {result[:100]}...")
        return result

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """处理一个batch的数据"""
        batch_size = len(instances)
        
        # 1. 处理对话文本
        conversations = [self._build_conversation_text(inst) for inst in instances]
        
        from llava.mm_utils import tokenizer_image_token
        
        input_ids_list = []
        for conv in conversations:
            input_ids = tokenizer_image_token(
                conv,
                tokenizer=self.tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            )
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids_list.append(input_ids.squeeze(0))
        
        # 手动padding
        max_length = max(ids.shape[0] for ids in input_ids_list)
        padded_input_ids = []
        attention_masks = []
        labels_list = []
        
        for input_ids in input_ids_list:
            pad_length = max_length - input_ids.shape[0]
            
            # 创建attention_mask
            attention_mask = torch.ones(input_ids.shape[0], dtype=torch.long)
            
            # 创建labels (简单版本：所有token都作为目标)
            labels = input_ids.clone()
            
            if pad_length > 0:
                # 右侧padding
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), -100, dtype=labels.dtype)  # padding部分忽略
                ])
            
            padded_input_ids.append(input_ids)
            attention_masks.append(attention_mask)
            labels_list.append(labels)
        
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels_list)
        }
        
        print(f"📝 批量tokenization完成: input_ids shape: {batch['input_ids'].shape}")
    
        # 2. 处理图像数据 - 🔧 关键修复：使用正确的model_cfg
        image_feature_length = 0
        if 'image_path' in instances[0]:
            # 先收集所有图像
            pil_images = []
            for inst in instances:
                if inst.get('has_image', False):
                    image_path = inst['image_path']
                    image = Image.open(image_path).convert('RGB')
                else:
                    # 创建空白图像
                    image = Image.new('RGB', (336, 336), (255, 255, 255))
                pil_images.append(image)
            
            # 🔧 关键修复：使用正确的model_cfg
            if self.image_processor:
                try:
                    # 调用 process_images 函数，使用传入的model_cfg
                    processed_images = process_images(
                        images=pil_images,
                        image_processor=self.image_processor,
                        model_cfg=self.model_cfg  # 🔧 使用正确的model_cfg
                    )
                    
                    # process_images 返回的可能是 tensor 或 tensor 列表
                    if isinstance(processed_images, list):
                        batch['images'] = torch.stack(processed_images)
                    else:
                        batch['images'] = processed_images
                    
                    image_feature_length = 576  # LLaVA-1.5 的图像特征长度
                    print(f"🖼️ LLaVA process_images 处理成功，shape: {batch['images'].shape}")
                    
                except Exception as e:
                    print(f"❌ LLaVA process_images 失败: {e}")
                    # 备用方案：直接使用 image_processor
                    processed = self.image_processor(pil_images, return_tensors="pt")
                    batch['images'] = processed['pixel_values']
                    image_feature_length = 576
                    print(f"🖼️ 备用图像处理成功，shape: {batch['images'].shape}")

        # 3. 处理音频数据
        audio_feature_length = 0
        if 'audio_path' in instances[0]:
            audio_features = []
            for inst in instances:
                if inst.get('has_audio', False):
                    audio_path = inst['audio_path']
                    
                    if self.audio_processor:
                        try:
                            audio_feat = self.audio_processor(audio_path)
                            
                            if hasattr(audio_feat, 'data'):
                                audio_feat = audio_feat.data
                            elif isinstance(audio_feat, dict):
                                for key, value in audio_feat.items():
                                    if isinstance(value, torch.Tensor):
                                        audio_feat = value
                                        break
                            
                            if audio_feat.dim() == 3 and audio_feat.shape[0] == 1:
                                audio_feat = audio_feat.squeeze(0)
                            
                            audio_feature_length = audio_feat.shape[0]
                            
                        except Exception as e:
                            print(f"❌ 音频处理失败: {e}")
                            audio_feat = torch.zeros(64, 1280)
                            audio_feature_length = 64
                    else:
                        audio_feat = torch.zeros(64, 1280)
                        audio_feature_length = 64
                else:
                    audio_feat = torch.zeros(64, 1280)
                    audio_feature_length = 64
                    
                audio_features.append(audio_feat)
            
            if audio_features:
                batch['audio_features'] = torch.stack(audio_features)
                print(f"🎵 Final audio batch shape: {batch['audio_features'].shape}")

        # 4. 输出统计信息
        text_length = batch["input_ids"].shape[1]
        total_length = text_length + image_feature_length + audio_feature_length
        
        print(f"📏 序列长度: 文本={text_length}, 图像={image_feature_length}, 音频={audio_feature_length}, 总计={total_length}")
        print(f"📏 Labels shape: {batch['labels'].shape}")

        return batch