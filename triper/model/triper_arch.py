import torch
import torch.nn as nn
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import IGNORE_INDEX
from triper.constants import AUDIO_TOKEN_INDEX
from .encoders.audio_encoder import build_audio_encoder
from .projectors.builder import build_audio_projector


class TriperMetaModel(LlavaMetaModel):
    """扩展LlavaMetaModel，添加音频支持"""
    
    def __init__(self, config):
        # 先初始化父类（视觉+文本）
        super(TriperMetaModel, self).__init__(config)
        
        # 添加音频模块
        if hasattr(config, "mm_audio_encoder"):
            self.audio_encoder = build_audio_encoder()
            self.audio_projector = build_audio_projector(config)

    def get_audio_encoder(self):
        """获取音频编码器"""
        audio_encoder = getattr(self, 'audio_encoder', None)
        if type(audio_encoder) is list:
            audio_encoder = audio_encoder[0]
        return audio_encoder

    def initialize_audio_modules(self, model_args, fsdp=None):
        """初始化音频模块（类似initialize_vision_modules）"""
        audio_encoder = model_args.audio_encoder
        pretrain_audio_projector = getattr(model_args, 'pretrain_audio_projector', None)
        
        self.config.mm_audio_encoder = audio_encoder
        
        if self.get_audio_encoder() is None:
            audio_encoder = build_audio_encoder(model_args)
            
            if fsdp is not None and len(fsdp) > 0:
                self.audio_encoder = [audio_encoder]
            else:
                self.audio_encoder = audio_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                audio_encoder = self.audio_encoder[0]
            else:
                audio_encoder = self.audio_encoder
            audio_encoder.load_model()

        self.config.use_audio_proj = True
        self.config.audio_projector_type = getattr(model_args, 'audio_projector_type', 'linear')
        self.config.audio_hidden_size = audio_encoder.hidden_size

        if getattr(self, 'audio_projector', None) is None:
            self.audio_projector = build_audio_projector(self.config)
        else:
            # 确保投影层参数可训练
            for p in self.audio_projector.parameters():
                p.requires_grad = True

        if pretrain_audio_projector is not None:
            audio_projector_weights = torch.load(pretrain_audio_projector, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.audio_projector.load_state_dict(get_w(audio_projector_weights, 'audio_projector'))


class TriperMetaForCausalLM(LlavaMetaForCausalLM):
    """扩展LlavaMetaForCausalLM，添加音频处理能力"""

    def get_audio_encoder(self):
        return self.get_model().get_audio_encoder()

    def encode_audios(self, audios):
        """编码音频（类似encode_images）"""
        audio_features = self.get_model().get_audio_encoder()(audios)
        audio_features = self.get_model().audio_projector(audio_features.to(dtype=self.dtype))
        return audio_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, audios=None, image_sizes=None
    ):
        """扩展原方法，支持三模态输入"""
        
        # 1. 先处理视觉模态（调用父类方法）
        vision_tower = self.get_vision_tower()
        audio_encoder = self.get_audio_encoder()
        
        # 如果只有音频没有图像
        if vision_tower is None or images is None:
            if audio_encoder is not None and audios is not None:
                return self._prepare_audio_only_inputs(
                    input_ids, position_ids, attention_mask, past_key_values, labels, audios
                )
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # 2. 处理图像特征（复用父类逻辑）
        if type(images) is list or images.ndim == 5:
            # ... 复用父类的图像处理逻辑 ...
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            # 简化处理，使用flat模式
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        # 3. 处理音频特征
        audio_features = None
        if audio_encoder is not None and audios is not None:
            if type(audios) is list:
                audios = [x.unsqueeze(0) if x.ndim == 2 else x for x in audios]
                concat_audios = torch.cat([audio for audio in audios], dim=0)
                audio_features = self.encode_audios(concat_audios)
                split_sizes = [audio.shape[0] for audio in audios]
                audio_features = torch.split(audio_features, split_sizes, dim=0)
                audio_features = [x.flatten(0, 1) for x in audio_features]
            else:
                audio_features = self.encode_audios(audios)

        # 4. 三模态融合处理
        return self._merge_multimodal_features(
            input_ids, position_ids, attention_mask, past_key_values, labels,
            image_features, audio_features
        )

    def _merge_multimodal_features(self, input_ids, position_ids, attention_mask, 
                                 past_key_values, labels, image_features, audio_features):
        """合并三模态特征"""
        
        # 处理None值
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 移除padding
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_audio_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 查找各模态token的位置
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_audios = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            
            if num_images == 0 and num_audios == 0:
                # 纯文本情况
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds_1)
                new_labels.append(labels[batch_idx])
                continue

            # 获取所有特殊token的位置
            special_token_indices = self._get_special_token_indices(cur_input_ids)
            
            # 切分文本片段
            cur_input_ids_segments = []
            cur_labels_segments = []
            cur_labels = labels[batch_idx]
            
            for i in range(len(special_token_indices) - 1):
                start_idx = special_token_indices[i] + 1
                end_idx = special_token_indices[i + 1]
                cur_input_ids_segments.append(cur_input_ids[start_idx:end_idx])
                cur_labels_segments.append(cur_labels[start_idx:end_idx])

            # 构建新的embedding序列
            cur_new_input_embeds = []
            cur_new_labels = []
            
            segment_idx = 0
            for i, token_idx in enumerate(special_token_indices[1:-1], 1):  # 跳过首尾的-1标记
                token_type = cur_input_ids[token_idx].item()
                
                # 添加文本段
                if segment_idx < len(cur_input_ids_segments):
                    text_embed = self.get_model().embed_tokens(cur_input_ids_segments[segment_idx])
                    cur_new_input_embeds.append(text_embed)
                    cur_new_labels.append(cur_labels_segments[segment_idx])
                
                # 添加模态特征
                if token_type == IMAGE_TOKEN_INDEX:
                    if image_features is not None and cur_image_idx < len(image_features):
                        cur_image_features = image_features[cur_image_idx]
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_image_idx += 1
                        
                elif token_type == AUDIO_TOKEN_INDEX:
                    if audio_features is not None and cur_audio_idx < len(audio_features):
                        cur_audio_features = audio_features[cur_audio_idx]
                        cur_new_input_embeds.append(cur_audio_features)
                        cur_new_labels.append(torch.full((cur_audio_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_audio_idx += 1
                
                segment_idx += 1
            
            # 添加最后一个文本段
            if segment_idx < len(cur_input_ids_segments):
                text_embed = self.get_model().embed_tokens(cur_input_ids_segments[segment_idx])
                cur_new_input_embeds.append(text_embed)
                cur_new_labels.append(cur_labels_segments[segment_idx])

            # 合并当前样本的embeddings
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # 后续的padding和返回逻辑与父类相同
        return self._finalize_multimodal_inputs(
            new_input_embeds, new_labels, _labels, _position_ids, _attention_mask
        )

    def _get_special_token_indices(self, input_ids):
        """获取所有特殊token的位置"""
        image_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        audio_indices = torch.where(input_ids == AUDIO_TOKEN_INDEX)[0].tolist()
        
        # 合并并排序所有特殊token位置
        all_indices = [-1] + sorted(image_indices + audio_indices) + [input_ids.shape[0]]
        return all_indices

    def _finalize_multimodal_inputs(self, new_input_embeds, new_labels, _labels, _position_ids, _attention_mask):
        """最终化多模态输入（复用父类逻辑）"""
        # 截断到最大长度
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # 合并批次
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_input_embeds[0].device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=new_input_embeds[0].device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        return None, position_ids, attention_mask, None, new_input_embeds, new_labels_padded if _labels is not None else None