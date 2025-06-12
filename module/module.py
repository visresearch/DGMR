from typing import Optional

from functools import partial

import torch

from dinov2.models import vision_transformer as dino_vit


from clip.eva_clip.factory import get_model_config
from clip.eva_clip.model import CLIPVisionCfg, RMSnorm, LayerNorm, _build_vision_tower
from clip.eva_clip.eva_vit_model import EVAVisionTransformer
from clip.eva_clip import get_tokenizer as evaclip_tokenizer, create_model_and_transforms as create_eva_model_and_transform

from timm.models.deit import VisionTransformerDistilled
from timm.models.swin_transformer import SwinTransformer
from intern_vit.build import build_model as build_internVIT
from clip.open_clip import create_model_and_transforms as create_openclip_model_and_transforms
from intern_vit.clip_benchmark.clip_benchmark.models.internvl_c_pytorch import load_internvit_c_pytorch
# FusedLayerNorm = LayerNorm

def build_model_dinov2(args, img_size=224) -> torch.nn.Module:

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

        model = dino_vit.__dict__[args.arch](**vit_kwargs)

        if hasattr(args, 'block_ids'):
            if args.ffn_layer == "swiglufused":
                for i, bids in enumerate(args.block_ids):
                    for bid in bids:
                        model.blocks[bid].mlp.w12 = torch.nn.Linear(model.embed_dim, 2*args.pruned_hidden_sizes[i])

                        model.blocks[bid].mlp.w3 = torch.nn.Linear(args.pruned_hidden_sizes[i], model.embed_dim)

    return model

def prune_model(model, cfg):
    if isinstance(model, SwinTransformer):
        if 'block_ids' in cfg:
            nstages = len(model.layers)
            for stage_id in range(nstages):
                nblocks = len(model.layers[stage_id].blocks)
                for block_id in range(nblocks):
                    block = model.layers[stage_id].blocks[block_id]

                    block.mlp.fc1 = torch.nn.Linear(block.dim, cfg.mlp_ratio * block.dim)
                    block.mlp.fc2 = torch.nn.Linear(cfg.mlp_ratio * block.dim, block.dim)
    else:
        NotImplementedError('Unrecognized model')
    
    return model

def build_model_eva_clip(cfg):
    model, train_transforms, val_transforms = create_eva_model_and_transform(cfg.arch, cfg.pretrained,force_custom_clip=True)
    return model.visual, train_transforms, val_transforms


def build_model_internVIT(cfg):
    return build_internVIT(cfg)


def build_model_openclip(cfg):
    model, train_transforms, val_transforms = create_openclip_model_and_transforms(cfg.arch, cfg.pretrained)
    return model.visual, train_transforms, val_transforms


def build_whole_model_openclip(cfg):
    model, train_transforms, val_transforms = create_openclip_model_and_transforms(cfg.arch, cfg.pretrained)
    return model, train_transforms, val_transforms


if __name__ == '__main__':
    from omegaconf import OmegaConf
    # cfg = OmegaConf.load('configs/intern_vit/distill_internvit6b.yaml')
    # teacher = build_model_internVIT(cfg.teacher)
    # student = build_model_internVIT(cfg.student)
    # print(teacher)
    # print(student)

    cfg = OmegaConf.load('configs/open_clip/distill_openclip.yaml')
    teacher = build_model_openclip(cfg, is_teacher=True)
    student = build_model_openclip(cfg, is_teacher=False)
    print(teacher.state_dict().keys())
    print(student.state_dict().keys())

