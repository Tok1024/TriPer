#!/usr/bin/env python3
"""
调试LLaVA的prepare_inputs_labels_for_multimodal函数
用于找出图像特征没有被正确插入的原因
"""

import os
import sys
import torch
import types
from typing import Optional, List, Union, Tuple

# 添加项目路径
sys.path.append('/home/wly/szl_all_code/triper-project')

from triper.model import from_pretrained_components
from triper.data import TriperDataset, TriperDataCollator
from llava.constants import IMAGE_TOKEN_INDEX

def setup_environment():
    """设置环境"""
    torch.cuda.empty_cache()
    print("🚀 环境设置完成")

def load_models():
    """加载模型和组件"""
    print("📦 正在加载模型...")
    
    audio_config = {
        'mm_audio_encoder': 'whisper_vq',
        'audio_hidden_size': 1280,
        'audio_model_path': '/sda1/glm-4-voice-tokenizer',
        'audio_projector_type': 'mlp2x_gelu',
        'audio_projector_hidden_dim': 2048,
        'dropout': 0.1
    }
    
    tokenizer, triper_model, image_processor, context_len, audio_encoder = from_pretrained_components(
        llava_model_path="/sda1/llava-v1.5-13b",
        audio_encoder_path="/sda1/glm-4-voice-tokenizer",
        audio_projector_path=None,
        audio_config=audio_config,
        freeze_llava=True,
        device_map="cuda:3"
    )
    
    print("✅ 模型加载完成")
    triper_model.get_parameter_stats()
    
    return tokenizer, triper_model, image_processor, audio_encoder

def load_dataset(tokenizer, image_processor, audio_encoder, triper_model):
    """加载数据集"""
    print("📊 正在加载数据集...")
    
    dataset = TriperDataset(
        json_path='/home/wly/szl_all_code/triper-project/data/simple_data_20_samples.json',
        media_root_path='/home/wly/szl_all_code/triper-project/data',
        mode="raw"
    )
    
    collator = TriperDataCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        audio_processor=audio_encoder,
        model_cfg=triper_model.llava_model.config
    )
    
    print("✅ 数据集加载完成")
    return dataset, collator

def create_debug_function(original_func):
    """创建调试版本的prepare_inputs_labels_for_multimodal函数"""
    
    def debug_prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None):
        print("\n" + "="*60)
        print("🔍 进入prepare_inputs_labels_for_multimodal函数")
        print("="*60)
        
        print(f"📝 参数信息:")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  input_ids content: {input_ids}")
        print(f"  images shape: {images.shape if images is not None else None}")
        print(f"  attention_mask: {attention_mask}")
        print(f"  labels: {labels}")
        print(f"  image_sizes: {image_sizes}")
        print(f"  position_ids: {position_ids}")
        print(f"  past_key_values: {past_key_values}")
        
        # 检查图像token
        if images is not None:
            image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)
            print(f"\n🖼️ 图像token分析:")
            print(f"  IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
            print(f"  图像token位置: {image_token_indices}")
            print(f"  图像token数量: {len(image_token_indices[0]) if len(image_token_indices) > 0 else 0}")
            
            # 检查每个位置的token
            for batch_idx in range(input_ids.shape[0]):
                batch_tokens = input_ids[batch_idx]
                image_positions = torch.where(batch_tokens == IMAGE_TOKEN_INDEX)[0]
                print(f"  批次 {batch_idx} 图像token位置: {image_positions.tolist()}")
        
        # 检查模型状态
        print(f"\n🔧 模型状态:")
        print(f"  模型训练模式: {self.training}")
        print(f"  Vision tower存在: {hasattr(self, 'get_vision_tower') and self.get_vision_tower() is not None}")
        
        if hasattr(self, 'get_vision_tower') and self.get_vision_tower() is not None:
            vision_tower = self.get_vision_tower()
            print(f"  Vision tower已加载: {vision_tower.is_loaded}")
            print(f"  Vision tower设备: {next(vision_tower.parameters()).device}")
        
        # 设置断点进行交互式调试
        print(f"\n⚠️  即将进入调试模式...")
        print(f"调试提示:")
        print(f"  - 使用 'n' 下一行")
        print(f"  - 使用 's' 步入函数")
        print(f"  - 使用 'c' 继续执行")
        print(f"  - 使用 'p 变量名' 查看变量")
        print(f"  - 使用 'q' 退出调试")
        
        import pdb; pdb.set_trace()
        
        # 调用原始函数
        print(f"\n🔄 调用原始函数...")
        result = original_func(
            input_ids=input_ids,
            position_ids=position_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            images=images,
            image_sizes=image_sizes
        )
        
        # 分析结果
        print(f"\n📊 函数返回结果分析:")
        if isinstance(result, (list, tuple)) and len(result) >= 6:
            new_input_ids, new_position_ids, new_attention_mask, new_past_key_values, inputs_embeds, new_labels = result
            
            print(f"  new_input_ids: {new_input_ids.shape if new_input_ids is not None else None}")
            print(f"  inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else None}")
            print(f"  new_attention_mask: {new_attention_mask.shape if new_attention_mask is not None else None}")
            
            if inputs_embeds is not None:
                original_len = input_ids.shape[1]
                new_len = inputs_embeds.shape[1]
                print(f"  长度变化: {original_len} → {new_len}")
                
                if new_len > original_len:
                    print(f"  ✅ 图像特征已插入！增加了 {new_len - original_len} 个token")
                else:
                    print(f"  ❌ 图像特征未插入！长度未变化")
        
        print("="*60)
        return result
    
    return debug_prepare_inputs_labels_for_multimodal

def debug_single_sample(triper_model, dataset, collator):
    """调试单个样本"""
    print("\n🧪 开始调试单个样本...")
    
    # 准备数据
    single_sample = dataset[0]
    batch_result = collator([single_sample])
    batch_result = {k: v.to(triper_model.device) for k, v in batch_result.items()}
    
    input_ids = batch_result['input_ids']
    images = batch_result['images']
    
    print(f"📝 输入数据:")
    print(f"  样本结构: {single_sample.keys()}")
    print(f"  input_ids: {input_ids}")
    print(f"  images shape: {images.shape}")
    
    # 保存原始函数并替换为调试版本
    original_prepare_func = triper_model.llava_model.prepare_inputs_labels_for_multimodal
    debug_func = create_debug_function(original_prepare_func)
    
    triper_model.llava_model.prepare_inputs_labels_for_multimodal = types.MethodType(
        debug_func, 
        triper_model.llava_model
    )
    
    try:
        print(f"\n🚀 开始调用prepare_inputs_labels_for_multimodal...")
        with torch.no_grad():
            result = triper_model.llava_model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=images,
                image_sizes=None
            )
        
        print(f"\n✅ 调用完成!")
        if len(result) >= 5:
            inputs_embeds = result[4]
            print(f"  最终inputs_embeds shape: {inputs_embeds.shape if inputs_embeds is not None else None}")
            
    except Exception as e:
        print(f"\n❌ 调用失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 恢复原始函数
        triper_model.llava_model.prepare_inputs_labels_for_multimodal = original_prepare_func
        print(f"\n🔄 已恢复原始函数")

def test_triper_model(triper_model, dataset, collator):
    """测试完整的TriperModel"""
    print(f"\n🧪 测试完整的TriperModel...")
    
    single_sample = dataset[0]
    batch_result = collator([single_sample])
    batch_result = {k: v.to(triper_model.device) for k, v in batch_result.items()}
    
    try:
        with torch.no_grad():
            output = triper_model(
                input_ids=batch_result['input_ids'],
                images=batch_result['images'],
                audio_features=batch_result['audio_features']
            )
        
        print("✅ TriperModel推理成功！")
        print(f"  输出结构: {output.keys()}")
        print(f"  输出logits形状: {output['logits'].shape}")
        
    except RuntimeError as e:
        print(f"❌ TriperModel推理失败: {e}")
        if "device" in str(e).lower():
            print("这可能是多GPU模型的设备分布问题")

def main():
    """主函数"""
    print("🐛 LLaVA多模态函数调试器")
    print("="*60)
    
    # 1. 设置环境
    setup_environment()
    
    # 2. 加载模型
    tokenizer, triper_model, image_processor, audio_encoder = load_models()
    
    # 3. 加载数据集
    dataset, collator = load_dataset(tokenizer, image_processor, audio_encoder, triper_model)
    
    # 4. 调试单个样本
    debug_single_sample(triper_model, dataset, collator)
    
    # 5. 测试完整模型
    test_triper_model(triper_model, dataset, collator)
    
    print("\n🎉 调试完成！")

if __name__ == "__main__":
    main()