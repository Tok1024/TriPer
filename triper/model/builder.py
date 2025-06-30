import torch
from typing import Optional, Dict, Any, Tuple
from llava.model.builder import load_pretrained_model
from .triper_arch import TriperModel
from triper.configs.triper_config import TriperConfig
from .audio import build_audio_encoder
from .audio import build_audio_projector


def from_pretrained_components(
    llava_model_path: str,
    audio_encoder_path: Optional[str] = None,
    audio_projector_path: Optional[str] = None,
    audio_config: Optional[Dict] = None,
    freeze_llava: bool = True,
    device_map: str = "auto"
) -> Tuple[Any, TriperModel, Any, int, Any]:
    """
    从预训练组件构建Triper模型
    
    Returns:
        tokenizer, triper_model, image_processor, context_len, audio_encoder
    """
    
    print("🔄 Building Triper model from components...")
    print(f"   LLaVA model: {llava_model_path}")
    print(f"   Audio encoder: {audio_encoder_path}")
    print(f"   Audio projector: {'Built from config' if audio_projector_path is None else audio_projector_path}")
    print(f"   Freeze LLaVA: {freeze_llava}")
    
    # 1. 加载LLaVA模型
    print("🔄 Loading LLaVA model...")
    tokenizer, llava_model, image_processor, context_len = load_pretrained_model(
        model_path=llava_model_path,
        model_base=None,
        model_name=llava_model_path.split('/')[-1],
        device_map=device_map
    )
    
    # # 🔧 关键修复：确保图像token正确配置
    # print("🔄 Configuring image tokens...")
    # from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    
    # # 检查并添加图像token
    # if DEFAULT_IMAGE_TOKEN not in tokenizer.get_vocab():
    #     print(f"添加图像token: {DEFAULT_IMAGE_TOKEN}")
    #     tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN])
        
    #     # 调整模型的embedding层
    #     llava_model.resize_token_embeddings(len(tokenizer))
        
    #     # 更新IMAGE_TOKEN_INDEX
    #     new_image_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    #     import llava.constants
    #     import triper.constants
    #     llava.constants.IMAGE_TOKEN_INDEX = new_image_id
    #     triper.constants.IMAGE_TOKEN_INDEX = new_image_id
    #     print(f"更新IMAGE_TOKEN_INDEX为: {new_image_id}")
    # else:
    #     print(f"✅ 图像token已存在: {DEFAULT_IMAGE_TOKEN}")
    
    # 2. 构建音频编码器（外部组件）
    audio_encoder = None
    if audio_encoder_path and audio_config:
        try:
            print("🔄 Building audio encoder...")
            
            # 确保音频配置包含路径
            if 'audio_model_path' not in audio_config:
                audio_config['audio_model_path'] = audio_encoder_path
            
            # 创建临时配置对象来构建编码器
            from triper.configs.triper_config import TriperConfig
            temp_config = TriperConfig(**audio_config)
            
            audio_encoder = build_audio_encoder(temp_config)
            
            # 🔧 关键修复：将音频编码器移到GPU
            if audio_encoder is not None:
                # 获取主模型的设备
                main_device = next(llava_model.parameters()).device
                print(f"🔄 Moving audio encoder to device: {main_device}")
                
                audio_encoder = audio_encoder.to(main_device)
                
                # 冻结音频编码器参数
                for param in audio_encoder.parameters():
                    param.requires_grad = False
                print("🔒 Audio encoder parameters frozen")
                
                print(f"✅ Audio encoder built and moved to {main_device}: {type(audio_encoder).__name__}")
            
        except Exception as e:
            print(f"⚠️ Failed to build audio encoder: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. 创建Triper配置并将投影器也移到GPU
    print("🔄 Creating Triper model...")
    
    config_dict = {
        'model_type': 'triper',
        'freeze_llava': freeze_llava,
        'hidden_size': getattr(llava_model.config, 'hidden_size', 5120),
        'vocab_size': getattr(llava_model.config, 'vocab_size', 32000),
    }
    
    # 合并音频配置
    if audio_config:
        config_dict.update(audio_config)
    
    triper_config = TriperConfig(**config_dict)
    
    # 4. 创建Triper模型
    triper_model = TriperModel(triper_config)
    
    # 🔧 将Triper模型移到与LLaVA相同的设备
    main_device = next(llava_model.parameters()).device
    print(f"🔄 Moving Triper model to device: {main_device}")
    triper_model = triper_model.to(main_device)
    
    # 5. 附加组件
    triper_model.attach_llava_model(llava_model)
    
    if audio_encoder is not None:
        triper_model.set_audio_encoder(audio_encoder)
        
    triper_model.set_components(tokenizer, image_processor, context_len)
    
    print("✅ Triper model created successfully!")
    triper_model.print_model_summary()
    
    return tokenizer, triper_model, image_processor, context_len, audio_encoder


def _create_triper_config_from_llava(
    llava_config, 
    audio_config: Optional[Dict[str, Any]] = None,
    freeze_llava: bool = True,
    freeze_audio_encoder: bool = True
) -> TriperConfig:
    """从LLaVA配置创建Triper配置"""
    
    # 获取LLaVA配置字典
    if hasattr(llava_config, 'to_dict'):
        config_dict = llava_config.to_dict()
    else:
        # 如果没有to_dict方法，手动提取关键配置
        config_dict = {
            'hidden_size': getattr(llava_config, 'hidden_size', 4096),
            'vocab_size': getattr(llava_config, 'vocab_size', 32000),
            'num_attention_heads': getattr(llava_config, 'num_attention_heads', 32),
            'num_hidden_layers': getattr(llava_config, 'num_hidden_layers', 32),
            'intermediate_size': getattr(llava_config, 'intermediate_size', 11008),
        }
    
    # 修改模型类型
    config_dict['model_type'] = 'triper'
    config_dict['architectures'] = ['TriperModel']
    
    # 添加Triper特定配置
    config_dict.update({
        'freeze_llava': freeze_llava,
        'freeze_audio_encoder': freeze_audio_encoder,
    })
    
    # 添加音频相关配置
    default_audio_config = {
        'mm_audio_encoder': 'whisper_vq',
        'audio_hidden_size': 768,
        'use_audio_proj': True,
    }
    
    if audio_config:
        default_audio_config.update(audio_config)
    
    config_dict.update(default_audio_config)
    
    return TriperConfig(**config_dict)


