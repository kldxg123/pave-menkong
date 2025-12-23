#    Copyright 2023 Haotian Liu
#    Licensed under the Apache License, Version 2.0 (the "License");

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from libs.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.nn.init import trunc_normal_

class PaveMetaModel(ABC):

    def __init__(self, config):
        self.vision_tower = None
        self.mm_projector = None
        self.video_feat_projector = None 
        self.second_feat_projector = None

        if hasattr(config, "use_mm_proj"):
            self.use_mm_proj = config.use_mm_proj
        else:
            self.use_mm_proj = True

    def get_vision_tower(self):
        return self.vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        
        self.config.mm_vision_tower = vision_tower
        if self.vision_tower is None:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = 'mlp2x_gelu'
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        
        self.init_video_feat_connector(model_args)

    def init_video_feat_connector(self, config):
        input_dim = 896 
        
        if getattr(config, 'use_fast_feat', False):
            feat_type = getattr(config, 'fast_feat_type', 'unknown')
            
            conf_dim = getattr(config, 'mm_hidden_size', 896)
            if isinstance(conf_dim, int) and conf_dim > 0:
                input_dim = conf_dim
            
            if feat_type == 'lstm':
                self.video_feat_projector = nn.Sequential(
                    nn.LSTM(input_dim, config.hidden_size, batch_first=True),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )
            elif feat_type == 'linear':
                self.video_feat_projector = nn.Linear(input_dim, config.hidden_size)
            elif feat_type == 'audio':
                self.video_feat_projector = nn.Sequential(
                    nn.Linear(input_dim, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )
            else:
                self.video_feat_projector = nn.Linear(input_dim, config.hidden_size)
                
            for m in self.video_feat_projector.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        if getattr(config, 'use_second_sides', False):
             self.second_feat_projector = nn.Linear(896, config.hidden_size)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, video_feat=None, video_feat_fps=None, feat_frame_nums=None, 
        second_feat=None, second_feat_fps=None, second_feat_frame_nums=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

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

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            
        projector = self.get_model().mm_projector
        embedding_layer = self.get_model().embed_tokens

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

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
            
            # [FIXED LOGIC] Ensure embedding input doesn't have -200
            # Although branch 2 typically handles this by construction, robust checks are needed
            flat_ids = torch.cat(cur_input_ids_noim)
            cur_input_embeds = embedding_layer(flat_ids)
            
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
                            temp_feat = curr_video_feat.unsqueeze(0).permute(0, 2, 1) 
                            target_len = curr_image_feat.shape[0]
                            temp_feat = F.interpolate(temp_feat, size=target_len, mode='linear', align_corners=False)
                            curr_video_feat = temp_feat.permute(0, 2, 1).squeeze(0)
                        
                        combined_feat = curr_image_feat + curr_video_feat
                        
                        if second_feat is not None:
                             curr_second = second_feat[batch_idx]
                             if curr_second.shape[0] != combined_feat.shape[0]:
                                temp_s = curr_second.unsqueeze(0).permute(0, 2, 1)
                                temp_s = F.interpolate(temp_s, size=combined_feat.shape[0], mode='linear', align_corners=False)
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