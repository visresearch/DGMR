import argparse

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
from torch.distributed import init_process_group, destroy_process_group

import os
from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.metrics import AccuracyAveraging

from module.module import load_evaclip_state_dict, build_model_eva_clip, build_model_dinov2, \
    build_model_openclip, build_model_internVIT
from clip.eva_clip import create_visual_model_and_transforms as build_eva_clip

def get_args():
    parser = argparse.ArgumentParser("Zero-Shot Classifition")
    parser.add_argument("--frame", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--train_dataset_str", type=str)
    parser.add_argument("--val_dataset_str", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--mlp_ratio", type=str)

    parser.set_defaults(
        frame = 'eva_clip',
        config_file = './configs/eva/eval.yaml',
        train_dataset_str = 'ImageNet:split=TRAIN:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra',
        val_dataset_str = 'ImageNet:split=VAL:root=/public/ILSVRC2012:extra=/public/scccse/dinov2-extra',
        output_dir = './out/knn',
        # batch_size = 256,
        batch_size = 512,
        weight=None,
        arch=None,
        mlp_ratio=None,
    )

    args = parser.parse_args()

    return args


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def knn_eval(args):
    from omegaconf import OmegaConf

    ddp_setup()

    cfg = OmegaConf.load(args.config_file)
    if args.frame == 'eva_clip':
        if args.weight is not None:
            cfg.model.weight = args.weight
        if args.arch is not None:
            cfg.model.vision_cfg.arch = args.arch
        # model = build_model_eva_clip(cfg.model, use_pruned=False).half()
        model,_ = build_eva_clip(cfg.model.vision_cfg.arch, cfg.model.weight)
        model=model.half()
        load_evaclip_state_dict(model, cfg.model.weight)
        model.norm = torch.nn.Identity()
        model.head = torch.nn.Identity()
    elif args.frame == 'dinov2':
        if args.weight is not None:
            cfg.model.weight = args.weight
        model = build_model_dinov2(cfg.model, cfg.model.input_size)
        state_dict = torch.load(cfg.model.weight, map_location='cpu')
        incompatible_keys = model.load_state_dict(state_dict, strict=True)
        print(incompatible_keys)
    elif args.frame == "open_clip":
        if args.weight is not None:
            cfg.model.pretrained = args.weight
        cfg.model.weight = cfg.model.pretrained
        if args.arch is not None:
            cfg.model.arch = args.arch
        model, _, _ = build_model_openclip(cfg.model)
        # model.ln_post = torch.nn.Identity()
    elif args.frame == "internVIT":
        if args.weight is not None:
            cfg.model.intern_vit_6b.pretrained = args.weight
        cfg.model.weight = cfg.model.intern_vit_6b.pretrained
        if args.mlp_ratio is not None:
            cfg.model.intern_vit_6b.mlp_ratio = int(args.mlp_ratio)
        model = build_model_internVIT(cfg)
        model.norm = torch.nn.Identity()
        model.head = torch.nn.Identity()
        model = model.half()
    else:
        NotImplementedError(f'Not recognize {args.frame}')

    model.cuda()
    
    os.makedirs(args.output_dir, exist_ok=True)

    eval_knn_with_model(
        model=model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        nb_knn=[10, 20, 100, 200],
        temperature=0.07,
        autocast_dtype=torch.half,
        accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
        transform=None,
        gather_on_cpu=False,
        batch_size=args.batch_size,
        num_workers=8,
        n_per_class_list=[-1],
        n_tries=1,
        model_path=cfg.model.weight,
    )

    destroy_process_group()

    return



if __name__ == '__main__':
    args = get_args()
    knn_eval(args)

