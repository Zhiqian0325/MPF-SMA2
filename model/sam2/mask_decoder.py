# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from model.sam2.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            num_classes: int,  # 新增：指定类别数量
            transformer_dim: int,
            transformer: nn.Module,
            activation: Type[nn.Module] = nn.GELU,
            use_high_res_features: bool = True,
            feat_size: Tuple[int, int] = (16, 16)
    ) -> None:
        """
        Predicts masks given an image, using a transformer architecture.
        直接为每个类别输出一个mask。

        Arguments:
          num_classes (int): 类别数量，每个类别对应一个mask
          transformer_dim (int): transformer的通道维度
          transformer (nn.Module): 用于预测mask的transformer
          activation (nn.Module): 上采样mask时使用的激活函数
          iou_head_depth (int): 用于预测mask质量的MLP深度
          iou_head_hidden_dim (int): 用于预测mask质量的MLP隐藏层维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes  # 存储类别数量
        self.feat_size = feat_size
        self.image_pe = nn.Parameter(torch.zeros(1, transformer_dim, feat_size[0], feat_size[1]))
        # 为每个类别创建一个mask token
        self.mask_tokens = nn.Embedding(num_classes, transformer_dim)

        # 输出上采样层
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        # 每个类别对应的MLP
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(num_classes)
            ]
        )


    def forward(
            self,
            image_embeddings: torch.Tensor,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
    ):
        """
        Predict masks given image embeddings.

        Arguments:
          image_embeddings (torch.Tensor): 图像编码器输出的特征
          image_pe (torch.Tensor): 位置编码，形状与image_embeddings相同
          repeat_image (bool): 是否在batch维度上重复图像特征

        Returns:
          torch.Tensor: 预测的mask [batch_size, num_classes, H, W]
          torch.Tensor: 每个mask的IOU预测分数 [batch_size, num_classes]
        """
        image_pe = self.image_pe
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # 返回所有类别的mask和IOU分数
        return masks

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
    ):
        """Predicts masks. See 'forward' for more details."""
        # 获取mask tokens (每个类别一个)
        batch_size = image_embeddings.shape[0]
        output_tokens = self.mask_tokens.weight  # [num_classes, C]
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_classes, C]

        # 在batch维度上扩展
        tokens = output_tokens.expand(batch_size, -1, -1)  # [B, num_classes, c]

        # 在batch维度上重复图像特征（如果需要）
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        # 注意：不再添加dense prompt embeddings

        # 位置编码处理
        assert image_pe.size(0) == 1, "位置编码batch维度应为1"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 运行transformer
        hs, src = self.transformer(
            image_embedding=src,  # 图像特征 [B, C, H, W]
            image_pe=pos_src,  # 位置编码 [B, C, H, W]
            point_embedding=output_tokens  # 类别tokens [B, num_classes, C]
        )
        mask_tokens_out = hs  # [B, num_classes, c]

        # 上采样mask embeddings
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # 为每个类别生成mask
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, num_classes, c']
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)


        return masks

# class MaskRefiner(nn.Module):
#     def __init__(self, in_channels=1, hidden_dim=32):
#         super().__init__()
#         self.in_channels = in_channels
#         self.hidden_dim = hidden_dim
#
#         # 时间特征提取器 (3D卷积)
#         # 关键修复：使用正确的输入通道数
#         self.temporal_encoder = nn.Sequential(
#             nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
#             nn.InstanceNorm3d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
#             nn.InstanceNorm3d(hidden_dim * 2),
#             nn.ReLU(inplace=True)
#         )
#
#         # 掩码残差生成器
#         self.mask_refiner = nn.Sequential(
#             nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
#             nn.GroupNorm(4, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
#             nn.Tanh()  # 输出[-1, 1]范围的微调量
#         )
#
#         # 时间上下文加权融合
#         self.fusion_gate = nn.Sequential(
#             nn.Conv2d(hidden_dim * 2 * 3, hidden_dim, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         # 时序一致性损失系数
#         self.temporal_weight = nn.Parameter(torch.ones(1) * 0.5)
#
#     def forward(self, masks):
#         """
#         输入: masks [B, T, C, H, W]
#         输出: refined_masks [B, T, C, H, W]
#         """
#         B, T, C, H, W = masks.shape
#
#         # 1. 提取时空特征
#         # 关键修复：确保输入通道数正确
#         masks_3d = masks.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
#         temporal_features = self.temporal_encoder(masks_3d)  # [B, F_dim, T, H, W]
#
#         # 2. 为每帧生成残差调整量
#         refined_frames = []
#         for t in range(T):
#             # 获取当前帧及其上下文特征
#             current_feat = temporal_features[:, :, t, :, :]  # [B, F_dim, H, W]
#
#             # 处理边界情况
#             prev_feat = temporal_features[:, :, t - 1, :, :] if t > 0 else torch.zeros_like(current_feat)
#             next_feat = temporal_features[:, :, t + 1, :, :] if t < T - 1 else torch.zeros_like(current_feat)
#
#             # 时间上下文融合
#             context = torch.cat([prev_feat, current_feat, next_feat], dim=1)  # [B, F_dim*3, H, W]
#             gate = self.fusion_gate(context)  # [B, 1, H, W]
#
#             # 加权融合上下文特征
#             fused_feat = (1 - gate) * current_feat + gate * (prev_feat + next_feat) / 2
#
#             # 生成当前帧的残差调整量
#             residual = self.mask_refiner(fused_feat)  # [B, C, H, W]
#
#             # 应用调整量
#             current_mask = masks[:, t]  # [B, C, H, W]
#             adjusted_mask = current_mask + residual * self.temporal_weight
#
#             refined_frames.append(adjusted_mask)
#
#         # 组合所有帧
#         return torch.stack(refined_frames, dim=1)  # [B, T, C, H, W]

class MaskRefiner(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 使用3D卷积进行时空特征提取，输入和输出尺寸相同
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, in_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

    def forward(self, masks):
        """
        输入: masks [B, T, C, H, W]
        输出: refined_masks [B, T, C, H, W]
        """

        # 1. 提取时空特征（使用3D卷积）
        masks_3d = masks.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        refined_masks_3d = self.temporal_encoder(masks_3d)  # [B, C, T, H, W]

        # 2. 恢复输出形状
        refined_masks = refined_masks_3d.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        return refined_masks