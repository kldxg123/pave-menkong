import torch
import torch.nn as nn

# 方向二：跨路径的早期交互
class GuidedTemporalAggregator(nn.Module):
    def __init__(self, audio_dim, vision_dim, target_dim, hidden_size):
        super().__init__()
        # 1. 将音频特征投影到目标维度
        self.audio_proj = nn.Linear(audio_dim, target_dim)
        
        # 2. 将视觉特征也投影到目标维度，作为Key/Value
        self.vision_proj = nn.Linear(vision_dim, target_dim)

        # 3. 核心交叉注意力层
        # Query来自音频，Key/Value来自视觉
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=target_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # 4. 输出层，将聚合后的特征投影到LLM维度
        self.output_proj = nn.Linear(target_dim, hidden_size)
        # 将时间维度压缩到与慢路径一致 (32 * 169)
        self.temporal_pool = nn.AdaptiveAvgPool1d(32 * 169)

    def forward(self, audio_feat, vision_feat):
        # audio_feat: [B, T_audio, C_audio] e.g., [2, 512, 768]
        # vision_feat: [B, T_vision, S_vision, C_vision] e.g., [2, 32, 576, 1152]
        
        B, T, S, C = vision_feat.shape
        
        # 准备Key/Value：将视觉特征展平到时间维度
        vision_feat_flat = vision_feat.view(B, T * S, C) # -> [B, 18432, 1152]
        
        # 准备Query, Key, Value
        queries = self.audio_proj(audio_feat) # -> [B, 512, target_dim]
        keys = self.vision_proj(vision_feat_flat) # -> [B, 18432, target_dim]
        values = keys
        
        # 交叉注意力计算
        attended_feat, attn_weights = self.cross_attention(queries, keys, values) # -> [B, 512, target_dim]
        
        # 时间压缩到目标token数量
        attended_feat = attended_feat.permute(0, 2, 1) # -> [B, target_dim, 512]
        pooled_feat = self.temporal_pool(attended_feat) # -> [B, target_dim, 18432]
        pooled_feat = pooled_feat.view(B, 32, 169, -1) # -> [B, 32, 169, target_dim]
        
        # 投影到LLM维度
        output = self.output_proj(pooled_feat) # -> [B, 32, 169, hidden_size]
        
        # 为了可解释性，返回注意力权重
        return output, attn_weights


# 方向一：动态与自适应 Patch
class AdaptiveFusionGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024), # 中间层维度可以调整
            nn.GELU(),
            nn.Linear(1024, 2) # 输出两个权重
        )

    def forward(self, slow_feat, fast_feat):
        # slow_feat: [B, T, S, D]
        # fast_feat: [B, T, S, D]
        
        # 1. 计算全局平均特征
        global_slow = slow_feat.mean(dim=[1, 2]) # -> [B, D]
        global_fast = fast_feat.mean(dim=[1, 2]) # -> [B, D]
        
        # 2. 拼接
        combined = torch.cat([global_slow, global_fast], dim=-1) # -> [B, D*2]
        
        # 3. 通过 MLP
        logits = self.mlp(combined) # -> [B, 2]
        
        # 4. 计算权重
        weights = torch.softmax(logits, dim=-1) # -> [B, 2]
        
        return weights # 返回 [w1, w2]
