import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder as GLMWhisperVQEncoder

import torch.nn as nn

class AudioCompressor(nn.Module):
    """可学习的音频特征压缩器"""
    
    def __init__(self, input_dim, output_seq_len, hidden_dim=None):
        super().__init__()
        self.output_seq_len = output_seq_len
        hidden_dim = hidden_dim or input_dim
        
        # 注意力池化
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(output_seq_len, input_dim))
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            compressed: [batch, output_seq_len, input_dim]
        """
        batch_size = x.shape[0]
        
        # 扩展查询向量到batch维度
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, output_seq_len, input_dim]
        compressed, _ = self.attention_pool(queries, x, x)
        
        return compressed

class WhisperVQEncoder(nn.Module):
    """Triper的音频编码器 - 包装GLM的WhisperVQ模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 🔧 从配置获取参数
        self.hidden_size = getattr(config, 'audio_hidden_size', 768)
        self.model_path = getattr(config, 'audio_model_path', "/sda1/glm-4-voice-tokenizer")
        if getattr(config, 'use_audio_compressor', True):
            self.target_seq_len = getattr(config, 'audio_target_seq_len', 64)
            self.audio_compressor = AudioCompressor(
                input_dim=self.hidden_size,
                output_seq_len=self.target_seq_len
            )

        # 🎵 加载Whisper模型
        try:
            self.whisper_model = GLMWhisperVQEncoder.from_pretrained(self.model_path).eval()
            # 🔧 明确冻结所有参数
            for param in self.whisper_model.parameters():
                param.requires_grad = False
            
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_path)
            
            # 获取实际的隐藏维度
            self.actual_hidden_size = self.whisper_model.config.d_model
            
            print(f"✅ WhisperVQEncoder loaded from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            self.whisper_model = None
            self.feature_extractor = None
            self.actual_hidden_size = self.hidden_size
    
    def preprocess_audio(self, audio_path_or_tensor):
        """预处理音频数据"""
        if isinstance(audio_path_or_tensor, str):
            # 从文件路径加载
            audio, sample_rate = torchaudio.load(audio_path_or_tensor)
        else:
            # 直接使用tensor
            audio = audio_path_or_tensor
            sample_rate = 16000  # 假设已经是16kHz
        
        # 重采样到16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            audio = resampler(audio)
        
        # 转为单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return audio[0].numpy()  # 这里仍然返回numpy，因为feature_extractor需要
    
    def extract_features_from_audio(self, audio_path_or_tensor):
        """从音频提取特征"""
        if self.whisper_model is None or self.feature_extractor is None:
            raise ValueError("Whisper model not loaded")
        
        # 预处理音频
        audio_array = self.preprocess_audio(audio_path_or_tensor)
        
        # 提取mel特征
        features = self.feature_extractor(
            audio_array, 
            sampling_rate=16000,
            return_attention_mask=True, 
            return_tensors="pt"
        )
        
        # 🔧 确保特征在正确的设备上
        device = next(self.whisper_model.parameters()).device
        features = features.to(device)
        
        # 通过Whisper编码器
        with torch.no_grad():
            outputs = self.whisper_model(
                input_features=features.input_features,
                attention_mask=features.attention_mask,
                quantized_token_ids=None  # 获取连续特征
            )
            continuous_features = outputs.last_hidden_state

            # 确保输出在与模型相同的设备上
            if continuous_features.device != device:
                continuous_features = continuous_features.to(device)
        
        # 如果使用音频压缩器，则应用它
        if hasattr(self, 'audio_compressor'):
            continuous_features = self.audio_compressor(continuous_features)
        
        return continuous_features
    
    def forward(self, audio_input):
        """
        前向传播
        Args:
            audio_input: 可以是:
                - 音频文件路径 (str)
                - 预处理的音频特征 (torch.Tensor)
                - 已经提取的特征 (torch.Tensor, shape=[B, T, D])
        """
        if isinstance(audio_input, str):
            # 从音频文件提取特征
            features = self.extract_features_from_audio(audio_input)
            return features
        elif audio_input.dim() == 2:
            # 假设是预处理的音频信号，需要提取特征
            features = self.extract_features_from_audio(audio_input)
            return features
        elif audio_input.dim() == 3:
            # 假设已经是特征，确保在正确设备上
            device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
            if isinstance(device, torch.device):
                target_device = device
            else:
                target_device = next(self.whisper_model.parameters()).device
            
            if audio_input.device != target_device:
                audio_input = audio_input.to(target_device)
            

            return audio_input
        else:
            raise ValueError(f"Unsupported audio input shape: {audio_input.shape}")

    
# 构建函数保持不变
def build_audio_encoder(config):
    """构建音频编码器"""
    try:
        mm_audio_encoder = getattr(config, 'mm_audio_encoder', 'whisper_vq')
        
        
        if mm_audio_encoder == 'whisper_vq':
            encoder = WhisperVQEncoder(config)
            return encoder
        else:
            raise ValueError(f"Unknown audio encoder: {mm_audio_encoder}")
            
    except Exception as e:
        print(f"❌ Failed to build audio encoder: {e}")
        import traceback
        traceback.print_exc()
        return None
