import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    
    # 📊 参数统计
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
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Triper模型的前向传播，扩展LLaVA支持音频"""
        if not self.is_ready():
            raise RuntimeError("模型组件尚未完全配置")

        # 🔧 添加输入验证和修复
        if input_ids is not None:
            # 1. 检查token范围
            vocab_size = self.llava_model.config.vocab_size
            print(f"🔍 Token范围检查: vocab_size={vocab_size}, input_ids range=({input_ids.min()}, {input_ids.max()})")
            
            # 修复超出范围的token
            if input_ids.max() >= vocab_size:
                print(f"⚠️ 发现超出词汇表的token，截断到{vocab_size-1}")
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            
            # 2. 确保数据类型正确
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
    
        if attention_mask is not None:
            print(f"🔍 Attention mask检查: dtype={attention_mask.dtype}, 值域=({attention_mask.min()}, {attention_mask.max()})")
            
            # 1. 确保数据类型正确
            if attention_mask.dtype not in [torch.long, torch.int, torch.bool]:
                print(f"⚠️ 修复attention_mask dtype: {attention_mask.dtype} -> torch.long")
                attention_mask = attention_mask.long()
            
            # 2. 确保值在有效范围内[0, 1]
            if attention_mask.min() < 0 or attention_mask.max() > 1:
                print(f"⚠️ 修复attention_mask值域: ({attention_mask.min()}, {attention_mask.max()}) -> [0, 1]")
                attention_mask = torch.clamp(attention_mask, 0, 1)
            
            # 3. 检查NaN/Inf
            if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
                print("⚠️ 检测到NaN/Inf，重置attention_mask...")
                attention_mask = torch.ones_like(attention_mask)

        print(f"🔥 TriperModel.forward called:")
        print(f"  input_ids: {input_ids.shape if input_ids is not None else 'None'}")
        print(f"  images: {images.shape if images is not None else 'None'}")
        print(f"  audio_features: {audio_features.shape if audio_features is not None else 'None'}")
        print(f"  past_key_values: {len(past_key_values) if past_key_values else 0} layers")
        
        # 🎯 关键修复：如果input_ids为None但有past_key_values，说明是后续生成步骤
        if input_ids is None and past_key_values is not None and len(past_key_values) > 0:
            print("  ⚡ input_ids为None且有past_key_values，直接调用LLaVA...")
            return self.llava_model.__class__.forward(
                self.llava_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )

        # 🔧 确定是否需要处理多模态输入
        need_multimodal_processing = (images is not None) or (audio_features is not None)
        
        if inputs_embeds is None and input_ids is not None and need_multimodal_processing:
            # 有多模态输入，需要处理
            
            # 1. LLaVA处理图像
            if images is not None:
                print("  📸 LLaVA处理图像...")
                multimodal_result = self.llava_model.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
                )
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = multimodal_result
                print(f"  LLaVA处理后embeds: {inputs_embeds.shape}")
            else:
                # 没有图像，直接获取文本embeds
                inputs_embeds = self.llava_model.get_model().embed_tokens(input_ids)
                print(f"  纯文本embeds: {inputs_embeds.shape}")

            # 2. 插入音频特征（只在第一步且有input_ids时）
            if audio_features is not None:
                print("  🎵 插入音频特征...")
                inputs_embeds, attention_mask = self._insert_audio_features(
                    inputs_embeds, input_ids, audio_features, attention_mask
                )
                print(f"  合并后embeds: {inputs_embeds.shape}")
                print(f"  合并后attention_mask: {attention_mask.shape}")

            print(f"  🔍 最终验证:")
            print(f"    inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else None}")
            print(f"    attention_mask: {attention_mask.shape if attention_mask is not None else None}")
            
            # 🔧 关键修复：处理完多模态后，清空input_ids
            input_ids = None  # 确保只传递inputs_embeds
            
        elif inputs_embeds is None and input_ids is not None and not need_multimodal_processing:
            # 🔧 纯文本情况：直接传递input_ids，不生成inputs_embeds
            print("  📝 纯文本输入，直接传递input_ids...")
            pass  # 保持input_ids，不设置inputs_embeds

        # 3. 调用LLaVA进行前向传播
        return self.llava_model.forward(
            input_ids=input_ids,  # 多模态时为None，纯文本时为原值
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # 多模态时有值，纯文本时为None
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        ) # type: ignore


    def _insert_audio_features(self, inputs_embeds, input_ids, audio_features, attention_mask=None):
        """插入音频特征到序列末尾，同时更新attention_mask"""
        if audio_features is None:
            return inputs_embeds, attention_mask

        batch_size = inputs_embeds.shape[0]
        audio_seq_len = audio_features.shape[1]
        
        # 1. 确保attention_mask与当前inputs_embeds长度匹配
        if attention_mask is not None and attention_mask.shape[1] != inputs_embeds.shape[1]:
            # 动态调整到实际的embeds长度
            actual_text_len = inputs_embeds.shape[1]
            if attention_mask.shape[1] > actual_text_len:
                attention_mask = attention_mask[:, :actual_text_len]
            else:
                # 填充到实际长度
                padding = torch.ones(
                    (batch_size, actual_text_len - attention_mask.shape[1]),
                    dtype=attention_mask.dtype, device=inputs_embeds.device
                )
                attention_mask = torch.cat([attention_mask, padding], dim=1)
    
        # 投影音频特征
        audio_embeds = self.audio_projector(audio_features)
        
        # 类型对齐
        if audio_embeds.dtype != inputs_embeds.dtype:
            audio_embeds = audio_embeds.to(inputs_embeds.dtype)
        
        # 拼接特征
        combined_embeds = torch.cat([inputs_embeds, audio_embeds], dim=1)
        
        # 3. 扩展attention_mask以匹配最终长度
        audio_mask = torch.ones((batch_size, audio_embeds.shape[1]), dtype=attention_mask.dtype, device=inputs_embeds.device)
        final_attention_mask = torch.cat([attention_mask, audio_mask], dim=1)
        
        print(f"🎵 音频特征插入完成:")
        print(f"  原始embeds: {inputs_embeds.shape}")
        print(f"  音频embeds: {audio_embeds.shape}")
        print(f"  合并后embeds: {combined_embeds.shape}")
        print(f"  合并后attention_mask: {final_attention_mask.shape}")
        
        # 🔧 验证长度一致性
        assert combined_embeds.shape[1] == final_attention_mask.shape[1], \
            f"长度不匹配: embeds={combined_embeds.shape[1]}, mask={final_attention_mask.shape[1]}"
        
        return combined_embeds, final_attention_mask
    
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, 
                                      attention_mask=None, inputs_embeds=None, **kwargs):
        """
        🎯 核心方法：为生成过程准备输入
        这个方法控制generate循环，为每一步的解码准备输入
        """
        print(f"🔧 TriperModel.prepare_inputs_for_generation called:")
        print(f"  input_ids: {input_ids.shape if input_ids is not None else None}")
        print(f"  past_key_values: {len(past_key_values) if past_key_values else 0} layers")
        print(f"  inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else None}")
        
        # 如果不是第一步（已经有缓存的key/value），那么input_ids就只是最新生成的那个token
        if past_key_values:
            input_ids = input_ids[:, -1:]
            print(f"  🔄 后续步骤，input_ids截取为: {input_ids.shape}")
        
        # 调用LLaVA的prepare_inputs_for_generation，让它处理大部分的准备工作
        model_inputs = self.llava_model.prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds, 
            **kwargs
        )
        
        print(f"  ✅ LLaVA准备的inputs: {list(model_inputs.keys())}")
        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """🚀 简化版的Triper generate方法"""
        if not self.is_ready():
            raise RuntimeError("模型组件尚未完全配置，无法进行生成")
        
        print(f"🚀 TriperModel.generate called:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  images: {images.shape if images is not None else None}")
        print(f"  audio_features: {audio_features.shape if audio_features is not None else None}")

        # 🔧 如果没有音频，直接调用LLaVA的generate
        if audio_features is None:
            print("📝 无音频输入，直接使用LLaVA...")
            return self.llava_model.generate(
                inputs=input_ids,  # 注意：LLaVA期望的是inputs，不是input_ids
                images=images,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # 🎯 有音频的情况：手动准备inputs_embeds然后调用LLaVA
        print("🎵 检测到音频输入，准备多模态embeddings...")
        
        # 🔧 修复attention_mask长度问题
        if attention_mask is not None and input_ids is not None:
            if attention_mask.shape[1] != input_ids.shape[1]:
                print(f"⚠️ attention_mask长度不匹配，截取到文本长度")
                attention_mask = attention_mask[:, :input_ids.shape[1]]
        
        # 1. 准备多模态inputs_embeds
        if images is not None:
            print("📸 LLaVA处理图像...")
            multimodal_result = self.llava_model.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None, images, image_sizes
            )
            _, _, attention_mask, _, inputs_embeds, _ = multimodal_result
            print(f"LLaVA处理后embeds: {inputs_embeds.shape}")
        else:
            # 没有图像，直接获取文本embeds
            inputs_embeds = self.llava_model.get_model().embed_tokens(input_ids)
            print(f"纯文本embeds: {inputs_embeds.shape}")

        # 2. 集成音频
        print("🎵 集成音频特征...")
        inputs_embeds, attention_mask = self._insert_audio_features(
            inputs_embeds, input_ids, audio_features, attention_mask
        )
        print(f"最终embeds: {inputs_embeds.shape}")
        print(f"最终attention_mask: {attention_mask.shape}")

        # # 3. 直接调用LLaVA的generate，传入准备好的inputs_embeds
        
        print("🚀 调用LLaVA.generate with inputs_embeds...")
        from transformers import Llama4ForCausalLM
        return Llama4ForCausalLM.generate(
            self.llava_model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )