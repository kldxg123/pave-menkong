#    Copyright 2024 Haotian Liu
#    Licensed under the Apache License, Version 2.0 (the "License");

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# [FIX] 补全所有必要的 Transformers 组件导入
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..pave_arch import PaveMetaModel
from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

class PAVEQwen2Config(Qwen2Config):
    model_type = "pave_qwen2"

class PAVEQwen2Model(PaveMetaModel, Qwen2Model):
    config_class = PAVEQwen2Config
    def __init__(self, config: Qwen2Config):
        Qwen2Model.__init__(self, config)
        PaveMetaModel.__init__(self, config)

class PAVEQwen2ForCausalLM(Qwen2ForCausalLM, PaveMetaModel):
    config_class = PAVEQwen2Config

    def __init__(self, config):
        config._attn_implementation = "eager"
        Qwen2ForCausalLM.__init__(self, config)
        PaveMetaModel.__init__(self, config)
        
        self.model = PAVEQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        video_feat: Optional[torch.FloatTensor] = None,
        video_feat_fps: Optional[torch.FloatTensor] = None,
        feat_frame_nums: Optional[torch.FloatTensor] = None,
        second_feat: Optional[torch.FloatTensor] = None,
        second_feat_fps: Optional[torch.FloatTensor] = None,
        second_feat_frame_nums: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                inputs_embeds,
                position_ids,
                attention_mask,
                past_key_values,
                input_ids,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images,
                video_feat, video_feat_fps, feat_frame_nums,
                second_feat, second_feat_fps, second_feat_frame_nums
            )

        # [CRITICAL SAFETY]
        # 一旦我们生成了 inputs_embeds，强制将 input_ids 设为 None。
        # 这样 Base Model 就只会用 embedding，不会尝试去索引 input_ids (里面可能含有 -200)
        if inputs_embeds is not None:
            B, seq_len, _ = inputs_embeds.shape
            
            # 对齐 Attention Mask
            if attention_mask is not None and attention_mask.shape[1] != seq_len:
                new_mask = torch.ones((B, seq_len), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = new_mask
            
            # 对齐 Position IDs
            if position_ids is None or position_ids.shape[1] != seq_len:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).repeat(B, 1)
            
            # [Fix] 强制 position_ids 非负，防止 RoPE 崩溃
            if position_ids is not None:
                position_ids = position_ids.clamp(min=0)

            input_ids = None

        return super().forward(
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

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, video_feat=None, video_feat_fps=None, feat_frame_nums=None, 
        second_feat=None, second_feat_fps=None, second_feat_frame_nums=None
    ):
        vision_tower = self.get_vision_tower()
        embedding_layer = self.get_model().embed_tokens
        REAL_VOCAB_SIZE = embedding_layer.weight.shape[0]

        # === 1. 解码阶段 / 纯文本阶段 ===
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # [绝对防御] 在进入 Embedding 之前，清洗所有非法 ID
            if input_ids is not None:
                # 克隆一份，避免修改原 tensor 导致外部逻辑错误
                safe_input_ids = input_ids.clone()
                # 将 -200 或其他负数强制归零
                safe_input_ids[safe_input_ids < 0] = 0 
                # 将越界正数强制钳制
                safe_input_ids = torch.clamp(safe_input_ids, max=REAL_VOCAB_SIZE - 1)
                # 替换原 input_ids 指向
                input_ids = safe_input_ids
                
            # 计算位置编码 (使用 KV Cache 长度，最稳健)
            if past_key_values is not None and input_ids.shape[1] == 1:
                # past_key_values shape: [layers, 2, batch, num_heads, seq_len, head_dim]
                current_seq_len = past_key_values[0][0].shape[-2]
                position_ids = torch.tensor([[current_seq_len]], dtype=torch.long, device=input_ids.device)
                
                # 扩展 Mask
                if attention_mask is not None:
                    target_len = current_seq_len + 1
                    if attention_mask.shape[1] < target_len:
                        pad_len = target_len - attention_mask.shape[1]
                        padding = torch.ones((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
                        attention_mask = torch.cat([attention_mask, padding], dim=1)
            elif position_ids is None and attention_mask is not None:
                 # 回退逻辑，增加 clamp 保护
                 position_ids = (torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1).clamp(min=0)

            return None, position_ids, attention_mask, past_key_values, input_ids, labels

        # === 2. 多模态 Prefill 阶段 ===
        if video_feat is not None:
            if hasattr(self.get_model(), 'video_feat_projector') and self.get_model().video_feat_projector is not None:
                dtype = self.get_model().mm_projector.weight.dtype
                device = self.get_model().mm_projector.weight.device
                video_feat = video_feat.to(dtype=dtype, device=device)
                video_feat = self.get_model().video_feat_projector(video_feat)

        if second_feat is not None:
            if hasattr(self.get_model(), 'second_feat_projector') and self.get_model().second_feat_projector is not None:
                dtype = self.get_model().mm_projector.weight.dtype
                device = self.get_model().mm_projector.weight.device
                second_feat = second_feat.to(dtype=dtype, device=device)
                second_feat = self.get_model().second_feat_projector(second_feat)

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.get_model().get_vision_tower()(concat_images)
            image_features = self.get_model().mm_projector(image_features)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            # [场景 A] 无图像，纯文本
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                # 清洗输入
                safe_ids = cur_input_ids.clone()
                safe_ids[safe_ids < 0] = 0
                safe_ids = safe_ids.clamp(max=REAL_VOCAB_SIZE - 1)
                
                cur_input_embeds_1 = embedding_layer(safe_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # [场景 B] 混合模态
            # 这里的 cur_input_ids 依然含有 -200，用于确定分割位置
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                if labels is not None:
                    cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_input_ids_noim]
            
            # [关键] 统一清洗并 Embedding
            if len(cur_input_ids_noim) > 0:
                flat_ids = torch.cat(cur_input_ids_noim)
                
                # 创建影子变量进行 Embedding，绝对安全
                safe_flat_ids = flat_ids.clone()
                safe_flat_ids[safe_flat_ids < 0] = 0 # 将残留的负数转为 0
                safe_flat_ids = safe_flat_ids.clamp(max=REAL_VOCAB_SIZE - 1)
                
                cur_input_embeds = embedding_layer(safe_flat_ids)
            else:
                cur_input_embeds = torch.empty((0, embedding_layer.weight.shape[1]), 
                                             dtype=embedding_layer.weight.dtype, 
                                             device=embedding_layer.weight.device)
            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            new_input_embeds_curr = []
            new_labels_curr = []

            for i in range(num_images + 1):
                new_input_embeds_curr.append(cur_input_embeds_no_im[i])
                if labels is not None:
                    new_labels_curr.append(cur_labels_noim[i])
                
                if i < num_images:
                    curr_image_feat = image_features[cur_image_idx]
                    
                    if video_feat is not None:
                        curr_video_feat = video_feat[batch_idx] 
                        if curr_video_feat.shape[0] != curr_image_feat.shape[0]:
                            if curr_video_feat.ndim > 2:
                                curr_video_feat = curr_video_feat.flatten(2).permute(1, 0)
                            temp_feat = curr_video_feat.unsqueeze(0).permute(0, 2, 1) 
                            target_len = curr_image_feat.shape[0]
                            temp_feat = F.interpolate(temp_feat.float(), size=target_len, mode='linear', align_corners=False).to(curr_image_feat.dtype)
                            curr_video_feat = temp_feat.permute(0, 2, 1).squeeze(0)
                        
                        combined_feat = curr_image_feat + curr_video_feat
                        if second_feat is not None:
                             curr_second = second_feat[batch_idx]
                             if curr_second.ndim > 2:
                                 curr_second = curr_second.flatten(2).permute(1, 0)
                             if curr_second.shape[0] != combined_feat.shape[0]:
                                temp_s = curr_second.unsqueeze(0).permute(0, 2, 1)
                                temp_s = F.interpolate(temp_s.float(), size=combined_feat.shape[0], mode='linear', align_corners=False).to(combined_feat.dtype)
                                curr_second = temp_s.permute(0, 2, 1).squeeze(0)
                             combined_feat = combined_feat + curr_second
                        new_input_embeds_curr.append(combined_feat)
                    else:
                        new_input_embeds_curr.append(curr_image_feat)
                    cur_image_idx += 1

            cur_new_input_embeds = torch.cat(new_input_embeds_curr)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(new_labels_curr)
                new_labels.append(cur_new_labels)

        if labels is not None:
            return torch.stack(new_input_embeds), position_ids, attention_mask, past_key_values, None, torch.stack(new_labels)
        else:
            return torch.stack(new_input_embeds), position_ids, attention_mask, past_key_values, None, None

# 注册配置和模型
AutoConfig.register("pave_qwen2", PAVEQwen2Config)
AutoModelForCausalLM.register(PAVEQwen2Config, PAVEQwen2ForCausalLM)