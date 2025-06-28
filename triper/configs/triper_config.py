from transformers import LlamaConfig
from typing import Optional

class TriperConfig(LlamaConfig):
    """Triper模型配置类，继承自LlamaConfig并添加音频相关配置"""
    
    model_type = "triper"
    
    def __init__(
        self,
        # 音频相关配置
        mm_audio_encoder: Optional[str] = None,
        audio_model_path: Optional[str] = None,
        audio_hidden_size: int = 1280,
        audio_target_seq_len: int = 64,
        audio_projector_type: str = "mlp",
        audio_projector_hidden_dim: int = 2048,  # 🔧 添加缺少的配置
        freeze_audio_encoder: bool = True,
        use_audio_compressor: bool = True,
        use_audio_proj: bool = True,
        tune_audio_projector: bool = True,
        audio_compressor_hidden_dim: int = 1024,
        
        # 🔧 添加其他必要配置
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        
        # Llava相关配置
        freeze_llava: bool = True,
        llava_model_path: Optional[str] = None,
        
        # 继承父类配置
        **kwargs
    ):
        # 先调用父类初始化
        super().__init__(**kwargs)
        
        # 设置音频相关配置
        self.mm_audio_encoder = mm_audio_encoder
        self.audio_model_path = audio_model_path
        self.audio_hidden_size = audio_hidden_size
        self.audio_target_seq_len = audio_target_seq_len
        self.audio_projector_type = audio_projector_type
        self.audio_projector_hidden_dim = audio_projector_hidden_dim
        self.freeze_audio_encoder = freeze_audio_encoder
        self.use_audio_proj = use_audio_proj
        self.tune_audio_projector = tune_audio_projector
        self.audio_compressor_hidden_dim = audio_compressor_hidden_dim
        
        # 设置LLaVA相关配置
        self.freeze_llava = freeze_llava
        self.llava_model_path = llava_model_path
        
        # 设置其他配置
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        # 确保模型类型正确
        self.model_type = "triper"