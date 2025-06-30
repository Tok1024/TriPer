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
    model_cfg: Optional[object] = None
    max_length: int = 2048

    def _build_conversation_text(self, instance: Dict) -> str:
        """构建对话预测任务的LLaVA格式"""
        text_parts = []
        
        # 添加图像token
        if instance.get('has_image', False):
            text_parts.append(DEFAULT_IMAGE_TOKEN)
        
        conversation = instance.get('conversation', [])
        
        if conversation:
            # 🎯 策略1：给出部分对话，让模型续写
            dialogue_lines = []
            for i, turn in enumerate(conversation[:-1]):  # 除了最后一句
                speaker = turn.get('speaker', 'Person')
                text = turn.get('text', '')
                if text.strip():
                    dialogue_lines.append(f"{speaker}: {text}")
            
            # 最后一句作为目标预测
            target_turn = conversation[-1]
            target_speaker = target_turn.get('speaker', 'Person')
            
            if dialogue_lines:
                context = "\n".join(dialogue_lines)
                text_parts.append(f"USER: Based on this conversation context and what you see/hear:\n{context}\n\nWhat would {target_speaker} say next?")
            else:
                text_parts.append(f"USER: Based on what you see and hear in this scene, what would {target_speaker} say?")
        else:
            text_parts.append("USER: Based on what you see and hear, what conversation would happen in this scene?")
        
        text_parts.append("ASSISTANT:")
        
        result = "\n".join(text_parts)
        print(f"📝 对话预测格式:\n{result}")
        return result

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """处理一个batch的数据"""
        batch_size = len(instances)
        
        # 1. 文本处理（现有逻辑保持不变）
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
        
        # 🔧 严格验证和padding
        original_lengths = [ids.shape[0] for ids in input_ids_list]
        max_length = max(original_lengths)
        min_length = min(original_lengths)
        
        print(f"📝 原始文本长度范围: {min_length} - {max_length}")
        if max_length != min_length:
            print(f"⚠️ 文本长度不一致，将padding到: {max_length}")
        
        # 确保padding后所有样本长度完全一致
        padded_input_ids = []
        attention_masks = []
        labels_list = []
        
        pad_token_id = self.tokenizer.pad_token_id
        if not isinstance(pad_token_id, int):
            raise ValueError("tokenizer.pad_token_id must be set to an integer for padding.")
        for i, input_ids in enumerate(input_ids_list):
            original_len = input_ids.shape[0]
            pad_length = max_length - original_len
            
            # 创建attention_mask
            attention_mask = torch.ones(original_len, dtype=torch.long)
            
            # 创建labels
            labels = input_ids.clone()
            
            if pad_length > 0:
                # 右侧padding
                padded_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), -100, dtype=labels.dtype)
                ])
            else:
                padded_ids = input_ids
            
            # 🔧 最终验证长度
            assert padded_ids.shape[0] == max_length, f"样本 {i} padding失败: {padded_ids.shape[0]} != {max_length}"
            assert attention_mask.shape[0] == max_length, f"样本 {i} attention_mask长度错误"
            assert labels.shape[0] == max_length, f"样本 {i} labels长度错误"
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)
            labels_list.append(labels)
        
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels_list)
        }
        
        # 🔧 最终批量验证
        assert batch["input_ids"].shape[0] == batch_size, "批量大小不匹配"
        assert batch["input_ids"].shape[1] == max_length, f"文本序列长度不一致: {batch['input_ids'].shape[1]} != {max_length}"
        assert batch["attention_mask"].shape == batch["input_ids"].shape, "attention_mask形状不匹配"
        assert batch["labels"].shape == batch["input_ids"].shape, "labels形状不匹配"
        
        print(f"✅ 批量tokenization完成: input_ids shape: {batch['input_ids'].shape}")
        print(f"✅ 所有样本文本长度统一为: {max_length}")
    
        # 2. 🔧 固定长度的图像处理
        image_feature_length = 576  # LLaVA-1.5 固定长度
        if any('image_path' in inst for inst in instances):
            pil_images = []
            for inst in instances:
                if inst.get('has_image', False) and 'image_path' in inst:
                    try:
                        image = Image.open(inst['image_path']).convert('RGB')
                    except Exception as e:
                        print(f"❌ 加载图像失败 {inst.get('image_path', 'Unknown')}: {e}")
                        image = Image.new('RGB', (336, 336), (255, 255, 255))
                else:
                    # 创建空白图像 - 确保批量中所有样本都有图像
                    image = Image.new('RGB', (336, 336), (255, 255, 255))
                pil_images.append(image)
            
            if self.image_processor:
                try:
                    processed_images = process_images(
                        images=pil_images,
                        image_processor=self.image_processor,
                        model_cfg=self.model_cfg
                    )
                    
                    if isinstance(processed_images, list):
                        batch['images'] = torch.stack(processed_images)
                    else:
                        batch['images'] = processed_images
                    
                    print(f"✅ 图像处理成功: {batch['images'].shape}")
                    
                    # 🔧 验证图像批量的一致性
                    assert batch['images'].shape[0] == batch_size, f"图像批量大小不匹配: {batch['images'].shape[0]} != {batch_size}"
                    
                except Exception as e:
                    print(f"❌ 图像处理失败: {e}")
                    # 备用：创建空白图像张量
                    batch['images'] = torch.zeros(batch_size, 3, 336, 336)
            else:
                print("⚠️ 没有图像处理器，创建空白图像")
                batch['images'] = torch.zeros(batch_size, 3, 336, 336)

        # 3. 🔧 固定长度的音频处理
        audio_feature_length = 64  # 固定压缩长度
        if any('audio_path' in inst for inst in instances):
            audio_features = []
            
            for inst in instances:
                if inst.get('has_audio', False) and 'audio_path' in inst:
                    audio_path = inst['audio_path']
                    
                    if self.audio_processor:
                        try:
                            audio_feat = self.audio_processor(audio_path)
                            
                            # 处理不同的返回格式
                            if hasattr(audio_feat, 'data'):
                                audio_feat = audio_feat.data
                            elif isinstance(audio_feat, dict):
                                for key, value in audio_feat.items():
                                    if isinstance(value, torch.Tensor):
                                        audio_feat = value
                                        break
                            
                            # 确保维度正确
                            if audio_feat.dim() == 3 and audio_feat.shape[0] == 1:
                                audio_feat = audio_feat.squeeze(0)
                            
                            # 🔧 验证音频特征长度
                            expected_shape = (audio_feature_length, 1280)  # (64, 1280)
                            if audio_feat.shape != expected_shape:
                                print(f"⚠️ 音频特征形状不匹配: {audio_feat.shape} != {expected_shape}")
                                audio_feat = torch.zeros(*expected_shape)
                            
                        except Exception as e:
                            print(f"❌ 音频处理失败 {audio_path}: {e}")
                            audio_feat = torch.zeros(audio_feature_length, 1280)
                    else:
                        audio_feat = torch.zeros(audio_feature_length, 1280)
                else:
                    # 空白音频特征
                    audio_feat = torch.zeros(audio_feature_length, 1280)
                    
                audio_features.append(audio_feat)
            
            # 🔧 验证所有音频特征形状一致
            shapes = [feat.shape for feat in audio_features]
            if len(set(shapes)) > 1:
                print(f"❌ 音频特征形状不一致: {shapes}")
                # 强制统一
                audio_features = [torch.zeros(audio_feature_length, 1280) for _ in range(batch_size)]
            
            batch['audio_features'] = torch.stack(audio_features)
            print(f"✅ 音频批量处理完成: {batch['audio_features'].shape}")
            
            # 🔧 最终验证
            expected_audio_shape = (batch_size, audio_feature_length, 1280)
            assert batch['audio_features'].shape == expected_audio_shape, \
                f"音频批量形状错误: {batch['audio_features'].shape} != {expected_audio_shape}"

        # 在TriperDataCollator中只处理基本的文本对齐
        return {
            'input_ids': batch['input_ids'],  # [batch, text_len]
            'attention_mask': batch['attention_mask'],  # [batch, text_len] - 只对应文本
            'labels': batch['labels'],  # [batch, text_len]
            'images': batch.get('images'),  # 可选，根据需要返回
            'audio_features': batch.get('audio_features'),  # 可选，根据需要返回
            # 不再预计算多模态总长度
        } # type: ignore