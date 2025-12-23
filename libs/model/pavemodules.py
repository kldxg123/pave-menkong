# /home/app-ahr/PAVE/libs/model/pavemodules.py

import torch
import torch.nn as nn
from einops import rearrange


class GuidedTemporalAggregator(nn.Module):
    """
    “智能口袋”：使用交叉注意力，让音频特征在早期就与视觉特征进行深度对话。
    """

    def __init__(self, audio_dim, vision_dim, target_dim, num_heads=8):
        super().__init__()
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        self.target_dim = target_dim
        self.num_heads = num_heads

        # 确保维度可以被头数整除
        assert target_dim % num_heads == 0

        # 将音频特征投影为Query
        self.audio_to_q = nn.Linear(audio_dim, target_dim)
        # 将视觉特征投影为Key和Value
        self.vision_to_kv = nn.Linear(vision_dim, 2 * target_dim)

        # 输出投影层
        self.proj = nn.Linear(target_dim, target_dim)

    def forward(self, audio_feat, vision_feat):
        """
        audio_feat: [B, T_a, C_a] 或 [B, T_a, H, W, C_a]
        vision_feat: [B, T_v, H, W, C_v]
        """
        # 如果音频特征是4D的 (例如从(C, T, 1, 1)转来)，先把它变成3D
        if audio_feat.dim() == 5:
            B, T_a, H, W, C_a = audio_feat.shape
            assert H == 1 and W == 1
            audio_feat = audio_feat.squeeze(2).squeeze(2)  # -> [B, T_a, C_a]

        # 如果视觉特征是5D的，先把它变成3D
        if vision_feat.dim() == 5:
            B, T_v, H, W, C_v = vision_feat.shape
            vision_feat = vision_feat.view(B, T_v, H * W, C_v)

        # 1. 准备 Q, K, V
        Q = self.audio_to_q(audio_feat)  # [B, T_a, target_dim]
        KV = self.vision_to_kv(vision_feat)  # [B, T_v*H*W, 2*target_dim]
        K, V = torch.chunk(KV, 2, dim=-1)  # [B, T_v*H*W, target_dim] each

        # 2. 计算注意力得分
        # Q: [B, T_a, target_dim] -> [B, num_heads, T_a, target_dim/num_heads]
        # K: [B, T_v*H*W, target_dim] -> [B, num_heads, target_dim/num_heads, T_v*H*W]
        Q = rearrange(Q, 'b t (h d) -> b h t d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h d s', h=self.num_heads)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K) / (self.target_dim / self.num_heads) ** 0.5  # [B, h, T_a, T_v*H*W]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # 3. 加权求和得到Value
        # V: [B, T_v*H*W, target_dim] -> [B, num_heads, T_v*H*W, target_dim/num_heads]
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)

        attended = torch.matmul(attn_weights, V)  # [B, h, T_a, target_dim/num_heads]
        attended = rearrange(attended, 'b h t d -> b t (h d)')  # [B, T_a, target_dim]

        # 4. 输出投影
        output = self.proj(attended)

        return output, attn_weights


class FiLMFusion(nn.Module):
    """
    “交通指挥中心”：使用FiLM动态地、通道级别地调制和融合两种特征。
    """

    def __init__(self, hidden_size):
        super().__init__()
        # FiLM生成器：根据两种特征的全局信息，生成gamma和beta
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * 4)  # 输出4个参数: gamma_slow, beta_slow, gamma_fast, beta_fast
        )

    def forward(self, slow_feat, fast_feat):
        # 1. 获取全局信息 (通过全局平均池化)
        # slow_feat: [B, T, S, D] -> [B, D]
        global_slow = slow_feat.mean(dim=[1, 2])
        # fast_feat: [B, T, D] -> [B, D]
        global_fast = fast_feat.mean(dim=[1, 2])

        # 2. 生成FiLM参数
        control_vector = torch.cat([global_slow, global_fast], dim=-1)  # [B, 2*D]
        gamma_beta = self.film_generator(control_vector)  # [B, 4*D]

        # 3. 拆分参数
        gamma_slow, beta_slow, gamma_fast, beta_fast = torch.chunk(gamma_beta, 4, dim=1)

        # 4. 调整形状以便广播
        # [B, D] -> [B, 1, 1, D]
        gamma_slow = gamma_slow.unsqueeze(1).unsqueeze(1)
        beta_slow = beta_slow.unsqueeze(1).unsqueeze(1)
        # [B, D] -> [B, 1, 1, D]
        gamma_fast = gamma_fast.unsqueeze(1).unsqueeze(1)
        beta_fast = beta_fast.unsqueeze(1).unsqueeze(1)

        # 5. FiLM调制
        modulated_slow = gamma_slow * slow_feat + beta_slow
        modulated_fast = gamma_fast * fast_feat + beta_fast

        # 6. 最终融合 (可以相加，也可以拼接，这里用相加)
        final_feat = modulated_slow + modulated_fast

        return final_feat
