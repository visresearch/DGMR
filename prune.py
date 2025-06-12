import argparse
import math
import os
from collections import OrderedDict
import sys
from typing import Tuple

from omegaconf import OmegaConf

import torch
import torch.nn as nn
from clip.eva_clip import create_model_and_transforms


import dinov2.models.vision_transformer as dinov2_archs

from module.module import build_model_openclip, build_model_eva_clip, build_internVIT
def eliminate(vec_in, vec_remove):

    vec_remove_normalize = vec_remove / torch.norm(vec_remove)

    dot = torch.dot(vec_in, vec_remove_normalize)

    vec_res = vec_in - dot * vec_remove_normalize

    return vec_res


def eliminate_batch(vecs, vec_remove):

    vec_remove_normalize = vec_remove / torch.norm(vec_remove)

    dot = torch.matmul(vecs, vec_remove_normalize).view(-1, 1)

    vecs_res = vecs - dot * vec_remove_normalize.view(1, -1)

    return vecs_res


def prune_diversity(weight):
    nrows, ncols = weight.shape

    idxs_row_select = []

    _weight = weight.clone()

    for i in range(ncols):
        norm = torch.norm(_weight, dim=1)
        norm_max, idx_row_select = torch.max(norm, dim=0)

        # print(norm_max)

        idxs_row_select.append(idx_row_select)

        _weight = eliminate_batch(_weight, _weight[idx_row_select])

        # print(torch.norm(_weight, dim=1))
    
    # print('the norm of residual weight', torch.norm(_weight))

    idxs_row_select = torch.LongTensor(idxs_row_select)

    return idxs_row_select


def prune_mlp_diversity(in_linear: nn.Linear, out_linear: nn.Linear):
    in_weight, in_bias = in_linear.weight.data, in_linear.bias.data
    out_weight, out_bias = out_linear.weight.data, out_linear.bias.data

    idxs_select = prune_diversity(in_weight)

    # --------------- input linear ---------------
    in_weight_prune = in_weight[idxs_select]
    in_bias_prune = in_bias[idxs_select] \
        if in_bias is not None else None

    in_linear_prune = nn.Linear(
        in_features=in_weight_prune.shape[1], 
        out_features=in_weight_prune.shape[0],
        bias=in_bias_prune is not None
    )

    in_linear_prune.weight.data = in_weight_prune
    in_linear_prune.bias.data = in_bias_prune

    # --------------- output linear ---------------
    out_weight_prune = out_weight[:, idxs_select]

    out_linear_prune = nn.Linear(
        in_features=out_weight_prune.shape[1], 
        out_features=out_weight_prune.shape[0],
        bias=out_bias is not None
    )

    out_linear_prune.weight.data = out_weight_prune
    out_linear_prune.bias.data = out_bias

    return in_linear_prune, out_linear_prune


def prune_diversity_kround(weight):
    nrows, ncols = weight.shape

    idxs_row_select = []

    _weight = weight.clone()

    for i in range(ncols):
        norm = torch.norm(_weight, dim=1)
        norm_max, idx_row_select = torch.max(norm, dim=0)

        idxs_row_select.append(idx_row_select)

        _weight = eliminate_batch(_weight, _weight[idx_row_select])
    
    is_select = [False for _ in range(nrows)]

    for idx in idxs_row_select:
        is_select[idx] = True
    
    idxs_row_unselect = []

    for i in range(nrows):
        if not is_select[i]:
            idxs_row_unselect.append(i)

    idxs_row_select = torch.LongTensor(idxs_row_select)
    idxs_row_unselect = torch.LongTensor(idxs_row_unselect)

    return idxs_row_select, idxs_row_unselect

def prune_mlp_diversity_kround(in_linear: nn.Linear, out_linear: nn.Linear, \
        mlp_factor: float=1, use_dual_prune: bool=False):

    in_weight, in_bias = in_linear.weight.data, in_linear.bias.data
    out_weight, out_bias = out_linear.weight.data, out_linear.bias.data

    nrows, ncols = in_weight.shape
    _in_weight = in_weight.clone()
    _out_weight = out_weight.clone()

    idxs_global_unselect = torch.arange(nrows)
    idxs_select = []

    for i in range(math.ceil(mlp_factor)):
        
        idxs_local_select, idxs_local_unselect = prune_diversity_kround(_in_weight)

        idxs_global_select = idxs_global_unselect[idxs_local_select]
        idxs_global_unselect = idxs_global_unselect[idxs_local_unselect]

        idxs_select.append(idxs_global_select)

        _in_weight = _in_weight[idxs_local_unselect]
        _out_weight = _out_weight[:, idxs_local_unselect]
    
    idxs_select = torch.cat(idxs_select, dim=0)
    out_size = int(in_linear.in_features*mlp_factor)
    idxs_select = idxs_select[:out_size]
    # --------------- input linear ---------------
    in_weight_prune = in_weight[idxs_select]
    in_bias_prune = in_bias[idxs_select] \
        if in_bias is not None else None

    in_linear_prune = nn.Linear(
        in_features=in_weight_prune.shape[1], 
        out_features=in_weight_prune.shape[0],
        bias=in_bias_prune is not None
    )

    in_linear_prune.weight.data = in_weight_prune
    in_linear_prune.bias.data = in_bias_prune

    # --------------- output linear ---------------
    out_weight_prune = out_weight[:, idxs_select]

    out_linear_prune = nn.Linear(
        in_features=out_weight_prune.shape[1], 
        out_features=out_weight_prune.shape[0],
        bias=out_bias is not None
    )

    out_linear_prune.weight.data = out_weight_prune
    out_linear_prune.bias.data = out_bias

    return in_linear_prune, out_linear_prune
def random_prune(in_linear: nn.Linear, out_linear: nn.Linear, mlp_ratio: float):
    in_weight, in_bias = in_linear.weight.data, in_linear.bias.data
    out_weight, out_bias = out_linear.weight.data, out_linear.bias.data



    # idxs_select = prune_diversity(in_weight)
    idxs_select = torch.torch.randint(0, in_linear.out_features, (int(mlp_ratio*in_linear.in_features),))

    # --------------- input linear ---------------
    in_weight_prune = in_weight[idxs_select]
    in_bias_prune = in_bias[idxs_select] \
        if in_bias is not None else None

    in_linear_prune = nn.Linear(
        in_features=in_weight_prune.shape[1], 
        out_features=in_weight_prune.shape[0],
        bias=in_bias_prune is not None
    )

    in_linear_prune.weight.data = in_weight_prune
    in_linear_prune.bias.data = in_bias_prune

    # --------------- output linear ---------------
    out_weight_prune = out_weight[:, idxs_select]

    out_linear_prune = nn.Linear(
        in_features=out_weight_prune.shape[1], 
        out_features=out_weight_prune.shape[0],
        bias=out_bias is not None
    )

    out_linear_prune.weight.data = out_weight_prune
    out_linear_prune.bias.data = out_bias

    return in_linear_prune, out_linear_prune

def l2_prune_mlp_weight(weight, pruned_hidden_size):
    norm = torch.linalg.norm(weight, dim=-1)

    value, indices = torch.topk(norm, pruned_hidden_size, largest=True, sorted=False)

    return indices


def l2_prune_mlp(in_linear: nn.Linear, out_linear: nn.Linear, \
        mlp_ratio:float) -> Tuple[nn.Linear, nn.Linear]:
    pruned_hidden_size = int(mlp_ratio * in_linear.in_features)
    in_weight, in_bias = in_linear.weight.clone(), in_linear.bias.clone()

    idxs_sample_selected = l2_prune_mlp_weight(in_weight, pruned_hidden_size)

    # --------------- input linear ---------------
    in_weight_prune = in_weight[idxs_sample_selected]
    in_bias_prune = in_bias[idxs_sample_selected] \
        if in_bias is not None else None

    in_linear_prune = nn.Linear(
        in_features=in_weight_prune.shape[1], 
        out_features=in_weight_prune.shape[0],
        bias=in_bias_prune is not None
    )

    in_linear_prune.weight.data = in_weight_prune
    in_linear_prune.bias.data = in_bias_prune

    # --------------- output linear ---------------
    out_weight, out_bias = out_linear.weight, out_linear.bias
    out_weight_prune = out_weight[:, idxs_sample_selected]

    out_linear_prune = nn.Linear(
        in_features=out_weight_prune.shape[1], 
        out_features=out_weight_prune.shape[0],
        bias=out_bias is not None
    )

    out_linear_prune.weight.data = out_weight_prune
    out_linear_prune.bias.data = out_bias


    return in_linear_prune, out_linear_prune

def taylor_prune_mlp(importance_path, model_save_path):
    cfg = OmegaConf.load('configs/open_clip/distill_openclip.yaml')
    model, _, _ = build_model_openclip(cfg.teacher)
    model.bfloat16()
    importances = torch.load(importance_path)
    with torch.no_grad():
        for idx, layer in enumerate(model.transformer.resblocks):
            layer.cuda()
            importance = importances[f"block_{idx}_mlp_c_fc"].cuda()
            dim = importance.shape[1]
            target_dim = dim * 1
            importance = torch.norm(importance, p=2, dim=0)
            _, select_indices = torch.topk(importance, k=target_dim, largest=True)

            fc1_weight_pruned = layer.mlp.c_fc.weight[select_indices,:] 
            fc1_bias_pruned = layer.mlp.c_fc.bias[select_indices]          
            fc2_weight_pruned = layer.mlp.c_proj.weight[:, select_indices]
            fc2_bias_pruned = layer.mlp.c_proj.bias

            layer.mlp.c_fc = torch.nn.Linear(dim, target_dim,dtype=torch.bfloat16)
            layer.mlp.c_proj = torch.nn.Linear(target_dim, dim,dtype=torch.bfloat16)

            layer.mlp.c_fc.weight.data = fc1_weight_pruned.data
            layer.mlp.c_fc.bias.data = fc1_bias_pruned.data
            layer.mlp.c_proj.weight.data = fc2_weight_pruned.data
            layer.mlp.c_proj.bias.data = fc2_bias_pruned.data
    print(model)
    torch.save(model.state_dict(), model_save_path)

def convert_opencip_whole():
    origin_ckpt = torch.load("/public/model_weight/CLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model.bin",map_location="cpu")
    prune_ckpt = torch.load("/public/model_weight/CLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model_taylor1.bin",map_location="cpu")
    for key in prune_ckpt.keys():
        full_key = "visual." + key
        origin_ckpt[full_key] = prune_ckpt[key]
    torch.save(origin_ckpt, "/public/model_weight/CLIP-ViT-g-14-laion2B-s34B-b88K/open_clip_pytorch_model_taylor1_whole.bin")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--prune_method', choices=['diversity', 'l2', 'random', 'taylor'])
    parser.add_argument('--arch', choices=['EVA02-CLIP-bigE-14-plus', 'EVA-CLIP-8B', 'ViT-g-14','ViT-bigG-14',"intern_vit_6b"])
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--mlp_ratio', type=float, help="must be between 0 and mlp_ratio of the original model")
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--importance_path', type=str)
    args = parser.parse_args()
    
    from module.module import build_internVIT, build_model_eva_clip, build_model_openclip
    if args.prune_method == "taylor":
        taylor_prune_mlp(args.importance_path, args.output_path)
        sys.exit()
    elif args.prune_method == "l2":
        prune_func = l2_prune_mlp
    elif args.prune_method == "random":
        prune_func = random_prune
    elif args.prune_method == "diversity":
        prune_func = prune_mlp_diversity_kround

    if args.arch.lower().startswith("eva"):
        model, _, _ = build_model_eva_clip(args)
        for idx, layer in model.blocks:
            layer.mlp.fc1, layer.mlp.fc2 = prune_func(layer.mlp.fc1, layer.mlp.fc2, args.mlp_ratio)
    elif args.arch.lower().startswith("intern"):
        cfg = OmegaConf.load("configs/intern_vit/knn_internvit6b.yaml")
        model = build_internVIT(cfg)
        for idx, layer in model.blocks:
            layer.mlp.fc1, layer.mlp.fc2 = prune_func(layer.mlp.fc1, layer.mlp.fc2, args.mlp_ratio)
    else:
        model, _, _ = build_model_openclip(args)
        for idx, layer in model.transformer.resblocks:
            layer.mlp.c_fc, layer.mlp.c_proj = prune_func(layer.mlp.c_fc, layer.mlp.c_proj, args.mlp_ratio)
    
    torch.save(model.state_dict(), args.output_path)
        
    
    
