"""
Model architecture extracted from dens_resi_mqxt.ipynb.
Input: RGB image tensor [B, 3, 224, 224]
Output: logits [B, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, kernel_size=3, dropout=0.0):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=kernel_size),
            nn.Dropout(dropout),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=kernel_size),
            nn.Dropout(dropout),
            ConvBNAct(hidden_channels, in_channels, kernel_size=kernel_size, act=False),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.block(x) + x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MQXABlock(nn.Module):
    """Multi-head Query-Key-Value Attention block."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, a, b):
        gate = self.gate(torch.cat([a, b], dim=1))
        return gate * a + (1.0 - gate) * b


class DenseNetResidualMQXA(nn.Module):
    """
    Shallow DenseNet121 backbone + Residual branch + MQXA attention branch + Gated Fusion.
    This must match the training notebook architecture exactly for weight loading.
    """
    def __init__(self, num_classes, token_dim=128, num_heads=4, dropout=0.10, use_pretrained_backbone=True):
        super().__init__()
        if use_pretrained_backbone:
            try:
                densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            except Exception:
                densenet = models.densenet121(pretrained=True)
        else:
            densenet = models.densenet121(weights=None)

        features_children = list(densenet.features.children())[:5]
        self.backbone = nn.Sequential(*features_children)

        backbone_out_dim = 256
        self.feat_reduce = ConvBNAct(backbone_out_dim, token_dim, kernel_size=1)
        self.residual_block = ResidualBlock(token_dim, hidden_channels=token_dim, dropout=dropout)
        self.mqxa_block = MQXABlock(token_dim, num_heads=num_heads, mlp_ratio=2.0, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fusion = GatedFusion(token_dim)
        self.dropout_final = nn.Dropout(dropout)
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.feat_reduce(features)

        residual_out = self.residual_block(features)
        residual_gap = self.gap(residual_out).flatten(1)

        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        mqxa_out = self.mqxa_block(tokens)
        mqxa_gap = mqxa_out.mean(dim=1)

        fused = self.fusion(residual_gap, mqxa_gap)
        logits = self.classifier(self.dropout_final(fused))
        return logits
