import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGate(nn.Module):
    """
    动态门控网络，根据输入特征的复杂度自适应生成权重
    """
    def __init__(self, hidden_dim=256):
        super(DynamicGate, self).__init__()
        # MLP网络用于生成权重
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入维度为2（视频和音频复杂度）
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # 输出两个权重
        )

    def compute_complexity(self, visual_feat, audio_feat):
        """
        计算视觉和音频特征的复杂度
        
        Args:
            visual_feat: 视觉特征 [B, T_v, D_v]
            audio_feat: 音频特征 [B, T_a, D_a]
            
        Returns:
            complexity: 复杂度向量 [B, 2]
        """
        # 计算视觉复杂度（相邻帧差异的均值）
        if visual_feat.size(1) > 1:
            visual_diff = torch.norm(visual_feat[:, 1:] - visual_feat[:, :-1], dim=2)  # [B, T_v-1]
            visual_complexity = torch.mean(visual_diff, dim=1, keepdim=True)  # [B, 1]
        else:
            visual_complexity = torch.zeros(visual_feat.size(0), 1, device=visual_feat.device)  # [B, 1]
            
        # 计算音频复杂度（能量均值）
        audio_energy = torch.norm(audio_feat, dim=2) ** 2  # [B, T_a]
        audio_complexity = torch.mean(audio_energy, dim=1, keepdim=True)  # [B, 1]
        
        # 拼接复杂度
        complexity = torch.cat([visual_complexity, audio_complexity], dim=1)  # [B, 2]
        return complexity
    
    def forward(self, visual_feat, audio_feat):
        """
        前向传播，计算动态权重
        
        Args:
            visual_feat: 视觉特征 [B, T_v, D_v]
            audio_feat: 音频特征 [B, T_a, D_a]
            
        Returns:
            weights: 归一化的权重 [B, 2]
        """
        # 计算复杂度
        complexity = self.compute_complexity(visual_feat, audio_feat)
        
        # 通过MLP生成权重
        weights = self.mlp(complexity)  # [B, 2]
        
        # 使用softmax归一化权重
        weights = F.softmax(weights, dim=1)  # [B, 2]
        
        return weights


class ModalFusion(nn.Module):
    """
    多模态特征融合模块
    """
    def __init__(self, visual_dim, audio_dim, target_dim, hidden_dim=256):
        super(ModalFusion, self).__init__()
        # 投影层，将不同模态特征投影到统一维度
        self.visual_proj = nn.Linear(visual_dim, target_dim)
        self.audio_proj = nn.Linear(audio_dim, target_dim)
        
        # 动态门控网络
        self.gate = DynamicGate(hidden_dim)
        
    def forward(self, visual_feat, audio_feat):
        """
        前向传播，融合视觉和音频特征
        
        Args:
            visual_feat: 视觉特征 [B, T_v, D_v]
            audio_feat: 音频特征 [B, T_a, D_a]
            
        Returns:
            fused_feat: 融合后的特征 [B, T, target_dim]
            weights: 使用的权重 [B, 2]
        """
        # 投影到统一维度
        visual_proj = self.visual_proj(visual_feat)  # [B, T_v, target_dim]
        audio_proj = self.audio_proj(audio_feat)     # [B, T_a, target_dim]
        
        # 计算动态权重
        weights = self.gate(visual_feat, audio_feat)  # [B, 2]
        
        # 平均时间维度以获得全局特征
        visual_global = torch.mean(visual_proj, dim=1)  # [B, target_dim]
        audio_global = torch.mean(audio_proj, dim=1)    # [B, target_dim]
        
        # 使用门控权重融合特征
        visual_weight = weights[:, 0:1]  # [B, 1]
        audio_weight = weights[:, 1:2]   # [B, 1]
        
        fused_feat = visual_weight * visual_global + audio_weight * audio_global  # [B, target_dim]
        
        return fused_feat, weights