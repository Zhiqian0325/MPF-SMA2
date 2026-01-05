import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .backbone.image_encoder import ImageEncoder
from .mask_decoder import MaskDecoder, MaskRefiner
from .transformer import TwoWayTransformer, RoPEAttention
from .sam2_utils import ObjPointer, get_1d_sine_pe
from .memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
from .memory_attention import MemoryAttention, MemoryAttentionLayer
from .position_encoding import PositionEmbeddingSine


class Sam2Refiner(nn.Module):
    def __init__(self, num_classes=3, image_size=256):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.num_feature_levels = 3
        self.hidden_dim = self.image_encoder.neck.d_model
        self.mem_dim = 64
        self.num_maskmem = 7
        self.num_classes = num_classes + 1
        self.use_high_res_features_in_sam = True
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=.02)
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.sam_mask_decoder = MaskDecoder(num_classes=self.num_classes,
                                            transformer_dim=self.hidden_dim,
                                            transformer=TwoWayTransformer(
                                                depth=2,
                                                embedding_dim=self.hidden_dim,
                                                mlp_dim=2048,
                                                num_heads=8,
                                            ),
                                            feat_size=(self.image_size[0] // 16, self.image_size[1] // 16), )
        self.obj_ptr_proj = ObjPointer(mlp_dim=(self.image_size[0] // 4) ** 2,
                                       hidden_dim=self.hidden_dim,
                                       num_classes=self.num_classes)
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=.02)
        self.use_obj_ptrs_in_encoder = True
        self.max_obj_ptrs_in_encoder = 16
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0
        self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.num_classes,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=self.num_classes,
                kernel_size=3,
                padding=1
            )
        )
        self.memory_attention = MemoryAttention(d_model=256,
                                                pos_enc_at_input=True,
                                                layer=MemoryAttentionLayer(activation='relu',
                                                                           dim_feedforward=2048,
                                                                           dropout=0.0,
                                                                           pos_enc_at_attn=False,
                                                                           self_attention=RoPEAttention(
                                                                               embedding_dim=256,
                                                                               num_heads=1,
                                                                               downsample_rate=1,
                                                                               dropout=0.0),
                                                                           d_model=256,
                                                                           pos_enc_at_cross_attn_keys=True,
                                                                           pos_enc_at_cross_attn_queries=False,
                                                                           cross_attention=RoPEAttention(
                                                                               embedding_dim=256,
                                                                               rope_k_repeat=True,
                                                                               num_heads=1,
                                                                               downsample_rate=1,
                                                                               dropout=0.0,
                                                                               kv_in_dim=64)),
                                                num_layers=4)
        self.memory_encoder = MemoryEncoder(out_dim=64,
                                            position_encoding=PositionEmbeddingSine(num_pos_feats=64),
                                            mask_downsampler=MaskDownSampler(kernel_size=3,
                                                                             stride=2,
                                                                             padding=1,
                                                                             num_classes=self.num_classes),
                                            fuser=Fuser(layer=CXBlock(dim=256),
                                                        num_layers=2))
        self.mask_refiner = MaskRefiner(self.num_classes)


    def forward(self, x):
        B, nF, C, H, W = x.shape
        x = x.flatten(0, 1)
        output_list = []
        backbone_out = self.forward_image(x)
        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

        imgs_ids = torch.arange(B * nF, device=x.device).reshape(B, nF).T
        for stage_id, img_ids in enumerate(imgs_ids):
            img_ids = img_ids.to(vision_feats[0].device)
            current_vision_feats = [vf[:, img_ids] for vf in vision_feats]
            current_vision_pos_embeds = [pe[:, img_ids] for pe in vision_pos_embeds]
            current_out = self.track_step(stage_id,
                                          current_vision_feats,
                                          current_vision_pos_embeds,
                                          feat_sizes,
                                          output_list,
                                          num_frames=nF
                                          )
            output_list.append(current_out)
        all_pred_masks_high_res = [output_list[key]['pred_masks_high_res'] for key in range(len(output_list))]
        all_pred_masks_high_res = torch.stack(all_pred_masks_high_res, dim=0).transpose(0, 1)
        all_pred_masks_high_res = self.mask_refiner(all_pred_masks_high_res)

        all_pred_masks_low_res = [output_list[key]['pred_masks'] for key in range(len(output_list))]
        all_pred_masks_low_res = torch.stack(all_pred_masks_low_res, dim=0).transpose(0, 1)
        return {'all_pred_masks_high_res': all_pred_masks_high_res, 'all_pred_masks_low_res': all_pred_masks_low_res}

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def track_step(self,
                   frame_idx,
                   current_vision_feats,
                   current_vision_pos_embeds,
                   feat_sizes,
                   output_list,
                   run_mem_encoder=True,
                   num_frames=8):

        current_out = {}

        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]
        pix_feat = self._prepare_memory_features(
            frame_idx=frame_idx,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_list=output_list,
            num_frames=num_frames
        )
        sam_outputs = self._forward_sam_heads(
            backbone_features=pix_feat,
            high_res_features=high_res_features,
        )
        (
            low_res_masks,
            high_res_masks,
            obj_ptr
        ) = sam_outputs
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            run_mem_encoder,
            high_res_masks,
            current_out,
        )
        return current_out

    def _prepare_memory_features(self,
                                 frame_idx,
                                 current_vision_feats,
                                 current_vision_pos_embeds,
                                 feat_sizes,
                                 output_list,
                                 num_frames=8):
        B = current_vision_feats[-1].size(1)  # batch size (8)
        C = self.hidden_dim  # 隐藏层维度 (256)
        mem_dim = self.mem_dim  # 记忆维度
        H, W = feat_sizes[-1]  # 特征图尺寸 (16x16)
        device = current_vision_feats[-1].device
        # 第一帧处理
        if frame_idx == 0:
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem

        to_cat_memory = []
        to_cat_memory_pos_embed = []
        num_obj_ptr_tokens = 0

        # 1. 收集视觉记忆特征
        memory_frame_indices = list(range(min(self.num_maskmem, len(output_list))))[::-1]

        for i, mem_idx in enumerate(memory_frame_indices):
            prev_output = output_list[mem_idx]
            if prev_output is not None:
                # 加载视觉记忆特征
                maskmem_features = prev_output["maskmem_features"].to(device, non_blocking=True)
                maskmem_pos_enc = prev_output["maskmem_pos_enc"][-1].to(device)

                # 处理视觉记忆特征为序列格式
                mem_feat_seq = maskmem_features.flatten(2).permute(2, 0, 1)

                # 处理位置编码 (空间+时间)
                spatial_pos = maskmem_pos_enc.flatten(2).permute(2, 0, 1)

                # 使用提供的时间位置编码格式 (num_maskmem, 1, 1, mem_dim)
                temporal_pos = self.maskmem_tpos_enc[i]  # [1, 1, mem_dim]

                # 扩展时间位置编码以匹配空间位置编码的形状
                temporal_pos = temporal_pos.expand(spatial_pos.size(0), B, mem_dim)

                # 合并空间和时间位置编码
                mem_pos_seq = spatial_pos + temporal_pos

                to_cat_memory.append(mem_feat_seq)
                to_cat_memory_pos_embed.append(mem_pos_seq)

        # 2. 处理对象指针
        if self.use_obj_ptrs_in_encoder:
            max_ptr_frames = min(num_frames, self.max_obj_ptrs_in_encoder)
            pos_and_ptrs = [
                # Temporal pos encoding contains how far away each pointer is from current frame
                (
                    (
                            frame_idx - t
                    ),
                    out["obj_ptr"],
                )
                for t, out in enumerate(output_list)
            ]
            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list = zip(*pos_and_ptrs)
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                t_diff_max = max_ptr_frames - 1
                tpos_dim = C
                obj_pos = torch.tensor(pos_list, device=device)
                obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                if self.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(
                        -1, B, C // self.mem_dim, self.mem_dim
                    )
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim * self.num_classes, dim=0)
                to_cat_memory.append(obj_ptrs)
                to_cat_memory_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0

        # 拼接所有记忆特征
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        fused_feat_seq = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens
        )

        # 重塑为空间特征图格式
        pix_feat_with_mem = fused_feat_seq.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem

    def _forward_sam_heads(
            self,
            backbone_features,
            high_res_features=None,
    ):
        """
        Forward SAM mask heads for multi-class mask prediction.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.

        Outputs:
        - low_res_masks: [B, num_classes, H*4, W*4] shape, the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_masks: [B, num_classes, H*16, W*16] shape,
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious: [B, num_classes] shape, the estimated IoU for each class mask
        """
        B = backbone_features.size(0)
        device = backbone_features.device

        # 直接调用修改后的多类别mask解码器
        low_res_masks = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # 转换mask为float32以确保兼容性
        low_res_masks = low_res_masks.float()

        # 上采样到原始图像尺寸
        high_res_masks = self.upsample_block(low_res_masks)

        # 为了兼容性，保留obj_ptr输出（但多类别情况下可能需要调整）
        # 这里简化为使用各类别mask的平均值作为代表性特征

        obj_ptr = self.obj_ptr_proj(low_res_masks.flatten(2))  # 输出形状 [B, num_classes, C]

        return (
            low_res_masks,  # [B, num_classes, H*4, W*4]
            high_res_masks,  # [B, num_classes, H*16, W*16]
            obj_ptr,  # 物体指针 [B, num_classes, C]
        )

    def _encode_memory_in_output(
            self,
            current_vision_feats,
            feat_sizes,
            run_mem_encoder,
            high_res_masks,
            current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def _encode_new_memory(
            self,
            current_vision_feats,
            feat_sizes,
            pred_masks_high_res,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        return maskmem_features, maskmem_pos_enc


