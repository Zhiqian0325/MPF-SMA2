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
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_classes = num_classes
        self.feat_size = feat_size
        self.image_pe = nn.Parameter(torch.zeros(1, transformer_dim, feat_size[0], feat_size[1]))

        self.mask_tokens = nn.Embedding(num_classes, transformer_dim)

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
        batch_size = image_embeddings.shape[0]
        output_tokens = self.mask_tokens.weight  # [num_classes, C]
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_classes, C]

        tokens = output_tokens.expand(batch_size, -1, -1)  # [B, num_classes, c]

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings

        assert image_pe.size(0) == 1, "位置编码batch维度应为1"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(
            image_embedding=src,
            image_pe=pos_src,
            point_embedding=output_tokens
        )
        mask_tokens_out = hs

        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, num_classes, c']
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)


        return masks

