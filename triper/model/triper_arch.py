import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from triper.constants import AUDIO_TOKEN_INDEX, IMAGE_TOKEN_INDEX
from .audio import build_audio_projector
from triper.configs.triper_config import TriperConfig


class TriperModel(PreTrainedModel):
    """
    Triper多模态模型 
    包含LLaVA作为视觉-语言子模块，并添加音频支持
    """
    config_class = TriperConfig
    
    def __init__(self, config: TriperConfig):
        super().__init__(config)
        self.config = config
        
        # 核心组件
        self.llava_model = None
        self.audio_projector = None  # 只有可训练的投影器
        
        # 私有引用的外部组件（不参与训练）
        self._tokenizer = None
        self._image_processor = None
        self._context_len = None
        self._audio_encoder = None  # 私有引用，不参与参数统计
        
        # 构建音频模块
        self._build_audio_modules()
        
        print(f"✅ TriperModel initialized with config: {config.model_type}")
        
    
    def _build_audio_modules(self):
        """构建音频模块 - 只构建可训练的投影器"""
        if hasattr(self.config, "mm_audio_encoder") and self.config.mm_audio_encoder:
            try:
                print(f"🔄 Building audio projector...")
                self.audio_projector = build_audio_projector(self.config)
                print(f"✅ Audio projector built: {type(self.audio_projector).__name__}")
                    
            except Exception as e:
                print(f"❌ Audio projector build failed: {e}")
                self.audio_projector = None
                raise
        else:
            print("⚠️ No audio encoder specified in config")
    
    def attach_llava_model(self, llava_model):
        """附加LLaVA模型作为子模块"""
        self.llava_model = llava_model
        print(f"✅ LLaVA model attached: {type(llava_model).__name__}")
        
        if hasattr(self.config, 'freeze_llava') and self.config.freeze_llava:
            self._freeze_llava_parameters()
    
    def set_audio_encoder(self, audio_encoder):
        """设置音频编码器（外部引用，不参与训练）"""
        self._audio_encoder = audio_encoder
        print(f"🎵 Audio encoder attached: {type(audio_encoder).__name__}")
    
    def set_components(self, tokenizer, image_processor, context_len):
        """一次性设置所有外部组件"""
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        self._context_len = context_len
        print(f"📦 Components set: tokenizer({type(tokenizer).__name__}), "
              f"processor({type(image_processor).__name__}), context_len({context_len})")
    
    def _freeze_llava_parameters(self):
        """冻结LLaVA模型参数"""
        if self.llava_model is not None:
            for param in self.llava_model.parameters():
                param.requires_grad = False
            print("🔒 LLaVA model parameters frozen")
    
    # 🎯 统一的组件访问器
    @property
    def tokenizer(self):
        """获取分词器"""
        return self._tokenizer
    
    @property
    def image_processor(self):
        """获取图像处理器"""
        return self._image_processor
    
    @property
    def context_len(self):
        """获取上下文长度"""
        return self._context_len or 2048
    
    @property
    def audio_encoder(self):
        """获取音频编码器（外部引用，不参与训练）"""
        return self._audio_encoder
    
    # 🔍 LLaVA子组件访问
    def get_model(self):
        """获取基础语言模型"""
        if self.llava_model is not None:
            return getattr(self.llava_model, 'model', None) or getattr(self.llava_model, 'get_model', lambda: None)()
        return None
    
    def get_vision_tower(self):
        """获取视觉塔"""
        return getattr(self.llava_model, 'get_vision_tower', lambda: None)() if self.llava_model else None
    
    def get_vision_projector(self):
        """获取视觉投影器"""
        if not self.llava_model:
            return None
        # 尝试多种可能的访问路径
        for attr_path in ['mm_projector', 'model.mm_projector']:
            obj = self.llava_model
            for attr in attr_path.split('.'):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None
    
    
    # 📊 参数统计（精简版）
    def get_parameter_stats(self) -> Dict[str, Any]:
        """获取参数统计信息 - 只统计可训练组件"""
        components = {
            'llava': self.llava_model,
            'audio_projector': self.audio_projector,
            # 注意：不包含 audio_encoder，因为它不参与训练
        }
        
        stats = {'total_params': 0, 'trainable_params': 0, 'components': {}}
        
        for name, component in components.items():
            comp_stats = {'total': 0, 'trainable': 0}
            if component is not None:
                for param in component.parameters():
                    comp_stats['total'] += param.numel()
                    if param.requires_grad:
                        comp_stats['trainable'] += param.numel()
            
            stats['components'][name] = comp_stats
            stats['total_params'] += comp_stats['total']
            stats['trainable_params'] += comp_stats['trainable']
        
        return stats
    
    def print_model_summary(self):
        """打印模型摘要"""
        print("\n🏗️  Triper Model Summary")
        print("=" * 60)
        
        # 组件状态
        print("📦 Components:")
        components_status = [
            ("🦙 LLaVA", self.llava_model),
            ("🎵 Audio Encoder", self._audio_encoder, "🔒 External (Frozen)"),
            ("🔗 Audio Projector", self.audio_projector, "🔓 Trainable"),
            ("📝 Tokenizer", self._tokenizer, "🔒 External"),
            ("🖼️ Image Processor", self._image_processor, "🔒 External")
        ]
        
        for item in components_status:
            if len(item) == 3:
                name, component, note = item
                status = "✅" if component is not None else "❌"
                type_info = f"({type(component).__name__}) {note}" if component else ""
            else:
                name, component = item
                status = "✅" if component is not None else "❌"
                type_info = f"({type(component).__name__})" if component else ""
            print(f"  {name}: {status} {type_info}")
        
        # 参数统计（只显示可训练参数）
        stats = self.get_parameter_stats()
        print(f"\n📊 Trainable Parameters:")
        print(f"  Total: {stats['total_params']:,}")
        print(f"  Trainable: {stats['trainable_params']:,} "
              f"({stats['trainable_params']/max(stats['total_params'], 1)*100:.1f}%)")
        
        # 组件详细统计
        for name, comp_stats in stats['components'].items():
            if comp_stats['total'] > 0:
                ratio = comp_stats['trainable'] / comp_stats['total'] * 100
                status = "🔓" if comp_stats['trainable'] > 0 else "🔒"
                print(f"    {name}: {comp_stats['total']:,} ({ratio:.1f}% trainable) {status}")
        
        print("=" * 60)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None, 
        inputs_embeds: Optional[torch.Tensor] = None,         
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.Tensor] = None,                
        image_sizes: Optional[List[List[int]]] = None,
        audio_features: Optional[torch.Tensor] = None,        
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Triper模型的前向传播，扩展LLaVA支持音频
        """
        if not self.is_ready():
            raise RuntimeError("模型组件尚未完全配置")
        

        print (f"初始input ids形状: {input_ids.shape if input_ids is not None else 'None'}")
        
        if inputs_embeds is None:
            # 1. 先处理图像（复用LLaVA的逻辑）
            if self.llava_model is None:
                raise RuntimeError("LLaVA model is not attached")
            
            print("images shape:", images.shape if images is not None else "None")
            
            multimodal_result = self.llava_model.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )
            
            

            # 明确类型转换
            input_ids = multimodal_result[0]
            position_ids = multimodal_result[1] 
            attention_mask = multimodal_result[2]
            past_key_values = multimodal_result[3]
            inputs_embeds = multimodal_result[4]
            labels = multimodal_result[5]
            
            print(f"处理图像后, embedding形状: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")
            
        # 2. 处理音频特征（新增逻辑）
        if audio_features is not None:
            inputs_embeds = self._insert_audio_features(
                inputs_embeds, input_ids, audio_features
            )
            
            print(f"插入音频后, 嵌入形状: {inputs_embeds.shape if inputs_embeds is not None else 'None'}")
        
        # 🔧 批量序列长度对齐 - 修复类型问题
        if inputs_embeds is not None and inputs_embeds.shape[0] > 1:  # 只在批量处理时执行
            print(f"🔍 批量长度对齐前:")
            print(f"  inputs_embeds shape: {inputs_embeds.shape}")
            print(f"  attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
            
            # 检查每个样本的实际长度
            if attention_mask is not None:
                actual_lengths = attention_mask.sum(dim=1)
                print(f"  实际长度: {actual_lengths.tolist()}")
                
                # 找到最大的实际长度 - 明确类型转换
                max_actual_length = int(actual_lengths.max().item())
                
                # 如果存在长度不一致，截断到最大实际长度
                if inputs_embeds.shape[1] != max_actual_length:
                    print(f"🔧 截断序列长度: {inputs_embeds.shape[1]} → {max_actual_length}")
                    
                    inputs_embeds = inputs_embeds[:, :max_actual_length, :]
                    attention_mask = attention_mask[:, :max_actual_length]
                    
                    if position_ids is not None:
                        position_ids = position_ids[:, :max_actual_length]
                    if labels is not None:
                        labels = labels[:, :max_actual_length]
                        
                    print(f"✅ 对齐后 inputs_embeds shape: {inputs_embeds.shape}")
            
            # 🔧 关键修复：确保所有样本的长度完全一致
            batch_size = inputs_embeds.shape[0]
            
            # 如果没有attention_mask，创建统一的attention_mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, inputs_embeds.shape[1]), 
                    dtype=torch.bool, 
                    device=inputs_embeds.device
                )
            
            # 检查是否所有样本长度都相同
            actual_lengths = attention_mask.sum(dim=1)
            unique_lengths = set(actual_lengths.tolist())
            
            if len(unique_lengths) > 1:
                # 如果长度不同，统一到最长的长度，进行padding而不是截断
                max_length = int(actual_lengths.max().item())  # 改为最大长度
                print(f"⚠️ 检测到不同长度，统一到最长长度: {max_length}")
                
                # 为每个样本进行padding
                batch_size = inputs_embeds.shape[0]
                embed_dim = inputs_embeds.shape[2]
                
                # 创建新的padded tensors
                new_inputs_embeds = torch.zeros(
                    (batch_size, max_length, embed_dim),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device
                )
                new_attention_mask = torch.zeros(
                    (batch_size, max_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                
                # 为每个样本填充数据
                for i in range(batch_size):
                    actual_len = int(actual_lengths[i].item())
                    # 复制实际数据
                    new_inputs_embeds[i, :actual_len, :] = inputs_embeds[i, :actual_len, :]
                    new_attention_mask[i, :actual_len] = 1
                    # 剩余部分自动为0 (padding)
                
                inputs_embeds = new_inputs_embeds
                attention_mask = new_attention_mask
                
                # 处理其他张量
                if position_ids is not None:
                    new_position_ids = torch.zeros(
                        (batch_size, max_length),
                        dtype=position_ids.dtype,
                        device=position_ids.device
                    )
                    for i in range(batch_size):
                        actual_len = int(actual_lengths[i].item())
                        if actual_len <= position_ids.shape[1]:
                            new_position_ids[i, :actual_len] = position_ids[i, :actual_len]
                        else:
                            # 如果position_ids较短，扩展它
                            new_position_ids[i, :position_ids.shape[1]] = position_ids[i]
                            new_position_ids[i, position_ids.shape[1]:actual_len] = torch.arange(
                                position_ids.shape[1], actual_len,
                                dtype=position_ids.dtype, device=position_ids.device
                            )
                    position_ids = new_position_ids
                
                if labels is not None:
                    new_labels = torch.full(
                        (batch_size, max_length),
                        -100,  # ignore_index for loss calculation
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    for i in range(batch_size):
                        actual_len = min(int(actual_lengths[i].item()), labels.shape[1])
                        new_labels[i, :actual_len] = labels[i, :actual_len]
                    labels = new_labels
                    
                print(f"🔧 最终padding后 inputs_embeds shape: {inputs_embeds.shape}")
                print(f"🔧 最终padding后 attention_mask shape: {attention_mask.shape}")
        
        # 3. 调用基础LLM
        if self.llava_model is None:
            raise RuntimeError("LLaVA model is not attached")
        
        return self.llava_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def _insert_audio_features(
        self, 
        inputs_embeds: Optional[torch.Tensor],  
        input_ids: Optional[torch.LongTensor], 
        audio_features: torch.Tensor          
    ) -> torch.Tensor:                        
        """插入音频特征到嵌入序列中"""

        # 处理音频特征
        if self._audio_encoder is None:
            raise RuntimeError("Audio encoder is not set")
        
        
        # 🔧 关键修复：确保音频编码器输出正确的数据类型
        with torch.no_grad():
            encoded_audio = self._audio_encoder(audio_features)
            
            # 确保编码后的音频特征类型正确
            if hasattr(self.llava_model, 'dtype'):
                target_dtype = self.llava_model.dtype
            else:
                # 从LLaVA模型的参数推断数据类型
                target_dtype = next(self.llava_model.parameters()).dtype
            
            if encoded_audio.dtype != target_dtype:
                print(f"  🔄 Converting encoded audio to {target_dtype}")
                encoded_audio = encoded_audio.to(dtype=target_dtype)
    
        if self.audio_projector is None:
            raise RuntimeError("Audio projector is not set")
        
        # 音频嵌入 - 投影器会自动处理数据类型匹配
        audio_embeds = self.audio_projector(encoded_audio)
        
        
        # 连接到 inputs_embeds 末尾
        if inputs_embeds is None:
            return audio_embeds
        else:
            print(f"  Inputs embeds shape: {inputs_embeds.shape}")
            print(f"  Inputs embeds dtype: {inputs_embeds.dtype}")
            
            # 🔧 确保数据类型匹配
            if audio_embeds.dtype != inputs_embeds.dtype:
                print(f"  🔄 Converting audio embeds to match inputs_embeds dtype")
                audio_embeds = audio_embeds.to(dtype=inputs_embeds.dtype)
            
            # 确保音频特征与输入嵌入的维度匹配
            if audio_embeds.size(-1) != inputs_embeds.size(-1):
                raise ValueError(
                    f"Audio features dimension ({audio_embeds.size(-1)}) "
                    f"does not match input embeddings dimension ({inputs_embeds.size(-1)})"
                )
            
            # 拼接张量
            result = torch.cat([inputs_embeds, audio_embeds], dim=1)
            print(f"  Final result shape: {result.shape}")
            print(f"  Final result dtype: {result.dtype}")
            return result

    # 🔍 便捷检查方法
    def is_ready(self) -> bool:
        """检查模型是否准备好推理"""
        return all([
            self.llava_model is not None,
            self._tokenizer is not None,
            self._image_processor is not None,
            self._audio_encoder is not None
        ])
        
    def to(self, device_or_dtype):
        """重写to方法，支持设备和数据类型转换"""
        # 移动主模型
        super().to(device_or_dtype)
        
        # 移动外部音频编码器
        if self._audio_encoder is not None:
            self._audio_encoder = self._audio_encoder.to(device_or_dtype)
            print(f"🔧 Audio encoder moved to: {device_or_dtype}")
        
        return self
    
    def cuda(self, device=None):
        """重写cuda方法"""
        return self.to(f'cuda:{device}' if device is not None else 'cuda')
    
    def cpu(self):
        """重写cpu方法"""
        return self.to('cpu')
    
    @property
    def device(self):
        """获取模型设备"""
        if self.llava_model is not None:
            return next(self.llava_model.parameters()).device
        return next(self.parameters()).device

