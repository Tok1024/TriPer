import torch
import torch.nn as nn


class AudioProjector(nn.Module):
    """音频特征投影器 - 将音频特征映射到LLM隐藏空间"""
    
    def __init__(self, config):
        super().__init__()
        
        audio_hidden_size = getattr(config, 'audio_hidden_size', 1280)
        hidden_size = getattr(config, 'hidden_size', 5120)
        projector_type = getattr(config, 'audio_projector_type', 'mlp2x_gelu')
        
        print(f"🔧 AudioProjector config:")
        print(f"  audio_hidden_size: {audio_hidden_size}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  projector_type: {projector_type}")
        
        # 支持多种投影器类型
        if projector_type == 'linear':
            self.projector = nn.Linear(audio_hidden_size, hidden_size)
        elif projector_type in ['mlp', 'mlp2x_gelu']:
            self.projector = nn.Sequential(
                nn.Linear(audio_hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )
        elif projector_type == 'mlp3x_gelu':
            projector_hidden_dim = getattr(config, 'audio_projector_hidden_dim', 2048)
            self.projector = nn.Sequential(
                nn.Linear(audio_hidden_size, projector_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(projector_hidden_dim, projector_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(projector_hidden_dim, hidden_size)
            )
        else:
            # 默认使用mlp2x_gelu
            print(f"⚠️ Unknown projector type '{projector_type}', using mlp2x_gelu")
            self.projector = nn.Sequential(
                nn.Linear(audio_hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self._init_weights()
        
        print(f"✅ AudioProjector created successfully")
    
    def _init_weights(self):
        """初始化权重"""
        try:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
        except Exception as e:
            print(f"⚠️ Weight initialization failed: {e}")
    
    def forward(self, audio_features):
        """
        Args:
            audio_features: [batch, seq_len, audio_hidden_size]
        Returns:
            projected: [batch, seq_len, hidden_size]
        """
        print(f"🎵 AudioProjector forward:")
        print(f"  Input shape: {audio_features.shape}")
        print(f"  Input dtype: {audio_features.dtype}")
        print(f"  Input device: {audio_features.device}")
        
        # 🔧 关键修复：确保数据类型匹配
        # 获取模型权重的数据类型
        model_dtype = next(self.parameters()).dtype
        print(f"  Model dtype: {model_dtype}")
        
        # 如果输入类型与模型类型不匹配，转换输入类型
        if audio_features.dtype != model_dtype:
            print(f"  🔄 Converting input from {audio_features.dtype} to {model_dtype}")
            audio_features = audio_features.to(dtype=model_dtype)
        
        # 确保设备匹配
        model_device = next(self.parameters()).device
        if audio_features.device != model_device:
            print(f"  🔄 Moving input from {audio_features.device} to {model_device}")
            audio_features = audio_features.to(device=model_device)
        
        try:
            projected = self.projector(audio_features)
            projected = self.layer_norm(projected)
            
            print(f"  Output shape: {projected.shape}")
            print(f"  Output dtype: {projected.dtype}")
            print(f"  Output device: {projected.device}")
            
            return projected
            
        except Exception as e:
            print(f"❌ AudioProjector forward failed: {e}")
            print(f"  projector type: {type(self.projector)}")
            raise
            
def build_audio_projector(config):
    """构建音频投影器"""
    try:
        projector = AudioProjector(config)
        return projector
        
    except Exception as e:
        print(f"❌ Failed to build audio projector: {e}")
        import traceback
        traceback.print_exc()
        return None
