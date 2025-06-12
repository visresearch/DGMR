# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(wd))

import logging

import torch

from . import vision_transformer as vits

from dinov2.layers.block_prune import PrunedBlock


logger = logging.getLogger("dinov2")

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

# def build_model_prune_mlp(args, only_teacher=False, img_size=224):
def build_model(args, only_teacher=False, img_size=224):
    # args.arch = args.arch.removesuffix("_memeff")
    args.arch = remove_suffix(args.arch, "_memeff")

    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)

        
        if hasattr(args, 'block_ids'):
            if args.ffn_layer == "swiglufused":
                for i, bids in enumerate(args.block_ids):
                    for bid in bids:
                        teacher.blocks[bid].mlp.w12 = torch.nn.Linear(teacher.embed_dim, 2*args.pruned_hidden_sizes[i])

                        teacher.blocks[bid].mlp.w3 = torch.nn.Linear(args.pruned_hidden_sizes[i], teacher.embed_dim)

        if only_teacher:
            return teacher, teacher.embed_dim

        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim

    return student, teacher, embed_dim


def build_model_prune_block(args, only_teacher=False, img_size=224):
# def build_model(args, only_teacher=False, img_size=224):
    # args.arch = args.arch.removesuffix("_memeff")
    args.arch = remove_suffix(args.arch, "_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)

        nblock = len(args.block_ids)

        if nblock > 0:
            if args.ffn_layer == "swiglufused":
                idxs_selected_neuron = torch.arange(args.pruned_hidden_size, dtype=torch.long)
                for i in range(nblock):
                    l = args.block_ids[i]
                    teacher.blocks[l] = PrunedBlock(
                        dim=teacher.blocks[l].mlp.w12.weight.shape[1],
                        num_heads=teacher.blocks[l].attn.num_heads,
                        idxs_selected_neuron=idxs_selected_neuron
                    )

        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim


    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)


def build_model_distill(args, img_size=224):

    args_s = args.student
    args_t = args.teacher

    args_s.arch = remove_suffix(args_s.arch, "_memeff")

    if "vit" in args_s.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args_s.patch_size,
            init_values=args_s.layerscale,
            ffn_layer=args_s.ffn_layer,
            block_chunks=args_s.block_chunks,
            qkv_bias=args_s.qkv_bias,
            proj_bias=args_s.proj_bias,
            ffn_bias=args_s.ffn_bias,
            num_register_tokens=args_s.num_register_tokens,
            interpolate_offset=args_s.interpolate_offset,
            interpolate_antialias=args_s.interpolate_antialias,
        )

        teacher = vits.__dict__[args_s.arch](**vit_kwargs)

        # logger.info(f'#block_chunks: {len(teacher.blocks)}')

        # block_chunks = args_s.block_chunks if hasattr(args_s, 'block_chunks') else len(teacher.blocks)
        block_chunks = len(teacher.blocks)
        chunk_size = teacher.n_blocks // block_chunks

        if hasattr(args_t, 'block_ids'):
            if args_s.ffn_layer == "swiglufused":
                if block_chunks < teacher.n_blocks:
                    for i, bids in enumerate(args_t.block_ids):
                        for bid in bids:
                            idx_chunk = bid // chunk_size
                            teacher.blocks[idx_chunk][bid].mlp.w12 = torch.nn.Linear(teacher.embed_dim, 2*args_t.pruned_hidden_sizes[i])
                            teacher.blocks[idx_chunk][bid].mlp.w3 = torch.nn.Linear(args_t.pruned_hidden_sizes[i], teacher.embed_dim)
                else:
                    for i, bids in enumerate(args_t.block_ids):
                        for bid in bids:
                            teacher.blocks[bid].mlp.w12 = torch.nn.Linear(teacher.embed_dim, 2*args_t.pruned_hidden_sizes[i])
                            teacher.blocks[bid].mlp.w3 = torch.nn.Linear(args_t.pruned_hidden_sizes[i], teacher.embed_dim)

        student = vits.__dict__[args_s.arch](
            **vit_kwargs,
            # drop_path_rate=args_s.drop_path_rate,
            # drop_path_uniform=args_s.drop_path_uniform,
        )

        if hasattr(args_s, 'block_ids'):
            if args_s.ffn_layer == "swiglufused":
                if block_chunks < teacher.n_blocks:
                    for i, bids in enumerate(args_s.block_ids):
                        for bid in bids:
                            idx_chunk = bid // chunk_size
                            student.blocks[idx_chunk][bid].mlp.w12 = torch.nn.Linear(student.embed_dim, 2*args_s.pruned_hidden_sizes[i])
                            student.blocks[idx_chunk][bid].mlp.w3 = torch.nn.Linear(args_s.pruned_hidden_sizes[i], student.embed_dim)
                else:
                    for i, bids in enumerate(args_s.block_ids):
                        for bid in bids:
                            student.blocks[bid].mlp.w12 = torch.nn.Linear(student.embed_dim, 2*args_s.pruned_hidden_sizes[i])
                            student.blocks[bid].mlp.w3 = torch.nn.Linear(args_s.pruned_hidden_sizes[i], student.embed_dim)
                            
        embed_dim = student.embed_dim

    return student, teacher, embed_dim


def build_model_from_cfg_distill(cfg):

    return build_model_distill(cfg, img_size=cfg.crops.global_crops_size)
