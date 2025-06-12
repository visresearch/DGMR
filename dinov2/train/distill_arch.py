# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
from torch import nn

import torch.nn.functional as F

from dinov2.models import build_model_from_cfg_distill
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk

from dinov2.convert_state_dict import convert_state_dict_inplace_naive2chunk

logger = logging.getLogger("dinov2")


class DistillArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg_distill(cfg)

        self.embed_dim = embed_dim
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.teacher.pretrained_weights:
            ckpt = torch.load(cfg.teacher.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.teacher.pretrained_weights}")
            ckpt = convert_state_dict_inplace_naive2chunk(ckpt, teacher_backbone.n_blocks, len(teacher_backbone.blocks))
            teacher_backbone.load_state_dict(ckpt, strict=True)
            # missing_keys, unexpected_keys = ', '.join(missing_keys), ', '.join(unexpected_keys)
            # logger.info(f'Missing keys: {missing_keys}')
            # logger.info(f'Unexpected keys: {unexpected_keys}')
        else:
            # logger.error(f"OPTIONS -- pretrained weights of teacher are missing!")
            logger.info(f"OPTIONS -- pretrained weights of teacher are missing!")

        if cfg.student.pretrained_weights:
            ckpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            ckpt = convert_state_dict_inplace_naive2chunk(ckpt, student_backbone.n_blocks, len(student_backbone.blocks))
            student_backbone.load_state_dict(ckpt, strict=True)
            # missing_keys, unexpected_keys = ', '.join(missing_keys), ', '.join(unexpected_keys)
            # logger.info(f'Missing keys: {missing_keys}')
            # logger.info(f'Unexpected keys: {unexpected_keys}')

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, x):

        with torch.no_grad():
            teacher_output_dict = self.teacher.backbone(x, is_training=True)

        # reshard_fsdp_model(self.teacher)

        student_output_dict = self.student.backbone(
            x, is_training=True
        )

        loss_dict = {}

        loss_cls = F.mse_loss(
            student_output_dict['x_norm_clstoken'],
            teacher_output_dict['x_norm_clstoken']
        )

        loss_patch = F.mse_loss(
            student_output_dict['x_norm_patchtokens'],
            teacher_output_dict['x_norm_patchtokens']
        )

        loss_dict['mse_loss_for_cls_tokens'] = loss_cls
        loss_dict['mse_loss_for_patch_tokens'] = loss_patch

        loss_accumulator = loss_cls + loss_patch

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()

            # self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def train(self):
        # super().train()
        self.student.train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k in self.student.keys():
            # self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
