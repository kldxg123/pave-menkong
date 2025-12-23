# /home/app-ahr/PAVE/libs/model/pave_qwen3vl_pathb.py

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from einops import rearrange

# 导入PAVE的基础架构和我们的新模块
from .pave_arch import PAVEMetaForCausalLM
from .pavemodules import GuidedTemporalAggregator, FiLMFusion
from .dynamic_gate import ModalFusion  # 导入动态门控模块


class PaveQwen3VLMPathB(PAVEMetaForCausalLM):
    """
    Path B 专用模型：
    - 使用Qwen3-VL的原生视觉编码器
    - 融合预计算的ImageBind音频特征
    """

    def __init__(self, config):
        super().__init__(config)

        # --- 初始化PAVE融合模块 ---
        # Qwen3-VL的视觉特征维度，动态获取
        qwen_vision_dim = self.config.vision_config.hidden_size
        # ImageBind的音频特征维度，根据常见设置，这里是768
        # TODO: 如果你的ImageBind音频特征维度不是768，请修改这里
        imagebind_audio_dim = 1024

        # 我们将所有特征都投影到Qwen的维度
        fusion_target_dim = qwen_vision_dim

        # 音频特征投影器：将ImageBind音频维度投影到Qwen视觉维度
        self.audio_projector = nn.Linear(imagebind_audio_dim, fusion_target_dim)

        # PAVE融合模块
        self.pave_aggregator = GuidedTemporalAggregator(
            audio_dim=fusion_target_dim,
            vision_dim=fusion_target_dim,
            target_dim=fusion_target_dim
        )
        self.pave_fusion = FiLMFusion(hidden_size=fusion_target_dim)
        
        # 动态门控融合模块
        self.modal_fusion = ModalFusion(
            visual_dim=qwen_vision_dim,  # 视觉特征维度
            audio_dim=fusion_target_dim,  # 音频特征维度
            target_dim=fusion_target_dim   # 目标维度
        )

        print(f"PAVE-Qwen3VL PathB model initialized.")
        print(f"  - Qwen Vision Dim: {qwen_vision_dim}")
        print(f"  - ImageBind Audio Dim: {imagebind_audio_dim}")
        print(f"  - Fusion Target Dim: {fusion_target_dim}")

    def prepare_inputs_labels_for_multimodal(
            self,
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes,
            modalities,
            video_metas,
            # ---- Path B 特有参数 ----
            video_feats,  # 这里实际上是音频特征
            video_feat_fps=None,
            feat_frame_nums=None,
    ):
        # 如果没有多模态输入，直接返回
        if images is None and video_feats is None:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # --- 1. 处理慢路径：原始视频帧 ---
        # images: list of [tensor], tensor shape: [T, 3, H, W]
        # 通过Qwen3-VL的视觉塔和投影器
        slow_path_feat = self.encode_images(images)  # -> list of [T, H*W, D]
        # 将list中的特征合并，并重塑为 [B, T, H, W, D]
        # 假设 batch_size=1, chunk_num=T
        slow_path_feat = torch.cat(slow_path_feat, dim=0)  # [T, H*W, D]
        T = images[0].shape[0]
        H = W = int(slow_path_feat.shape[1] ** 0.5)
        slow_path_feat = slow_path_feat.view(1, T, H, W, -1)  # [1, T, H, W, D]

        # --- 2. 处理快路径：音频特征 ---
        # video_feats: [B, C, T, 1, 1]
        fast_path_feat = video_feats.permute(0, 2, 3, 4, 1)  # [B, T, 1, 1, C]
        # 通过音频投影器，将维度C变为D
        fast_path_feat = self.audio_projector(fast_path_feat)  # [B, T, 1, 1, D]

        # --- 3. PAVE深度融合 ---
        # 3.1 早期交互：让音频特征"看"视觉特征
        attended_audio_feat, _ = self.pave_aggregator(
            fast_path_feat.squeeze(2).squeeze(2),  # [B, T, D]
            slow_path_feat  # [B, T, H, W, D]
        )
        # attended_audio_feat: [B, T, D], 我们把它变回 [B, T, 1, 1, D] 以便后续融合
        attended_audio_feat = attended_audio_feat.unsqueeze(2).unsqueeze(3)

        # 3.2 动态门控融合：使用动态门控网络进行特征融合
        # 展平视觉特征以适配门控网络
        flat_slow_path_feat = slow_path_feat.reshape(1, T, -1)  # [B, T, H*W*D]
        flat_attended_audio_feat = attended_audio_feat.squeeze(3).squeeze(2)  # [B, T, D]
        
        # 使用动态门控网络进行融合
        gated_fused_feat, gate_weights = self.modal_fusion(
            flat_slow_path_feat,  # [B, T, H*W*D]
            flat_attended_audio_feat  # [B, T, D]
        )
        # gated_fused_feat: [B, D]
        # gate_weights: [B, 2]
        
        # 将融合后的特征重新整形以适配后续处理
        gated_fused_feat = gated_fused_feat.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D]
        gated_fused_feat = gated_fused_feat.expand(-1, T, H*W, -1)  # [B, T, H*W, D]
        gated_fused_feat = gated_fused_feat.reshape(1, T, H, W, -1)  # [B, T, H, W, D]
        
        # 可选：打印门控权重用于调试
        # print(f"Gate weights: visual={gate_weights[0, 0].item():.3f}, audio={gate_weights[0, 1].item():.3f}")

        # 3.3 最终融合：使用FiLM进行最终融合
        final_multimodal_features = self.pave_fusion(slow_path_feat, gated_fused_feat)
        # final_multimodal_features: [B, T, H, W, D]

        # --- 4. 将融合后的特征嵌入到输入序列中 ---
        # 这部分逻辑与原始PAVE类似，但输入是我们的final_multimodal_features
        # 我们需要将其展平为 [B, T*H*W, D] 以便后续处理
        new_video_features = final_multimodal_features.view(1, -1, final_multimodal_features.shape[-1])
        new_frame_num = [new_video_features.shape[1]]

        # 接下来的代码与原始prepare_inputs_labels_for_multimodal的后半部分几乎一样
        # 只是把 video_features 替换为我们的 new_video_features
        # 为了简洁，我直接复制并修改了关键部分

        # (以下代码大量借鉴自原始pave_arch.py，并做了适配)
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

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in
                     zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == -200).sum()  # -200 is IMAGE_TOKEN_INDEX
            if num_images == 0:
                cur_image_features = new_video_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == -200)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = new_video_features[cur_image_idx]
                    cur_feature_len = new_frame_num[cur_image_idx]
                    cur_image_features = cur_image_features[:cur_feature_len]
                    if len(cur_image_features.shape) == 3:
                        cur_image_features = cur_image_features.view(-1, cur_image_features.shape[-1])
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device,
                                   dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype,
                                       device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                              device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype,
                                                             device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels