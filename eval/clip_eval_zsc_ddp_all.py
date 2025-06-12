import os
import argparse

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import json
import torch
from torch.distributed import init_process_group, destroy_process_group, get_rank

from intern_vit.clip_benchmark.clip_benchmark.models import load_clip as load_intern_vit
from clip.clip_benchmark.datasets.builder import (build_dataset, get_dataset_collate_fn)
from clip.clip_benchmark.models import load_clip, load_clip_visual
from clip.open_clip import create_model_and_transforms as openclip_model_and_transform, get_tokenizer as openclip_tokenizer
from eval import zeroshot_classification_ddp

def get_args():
    parser = argparse.ArgumentParser("Zero-Shot Classifition")
    parser.add_argument("--model", type=str)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--save_clf", type=str)
    parser.add_argument("--load_clfs", type=str, nargs='+')
    # parser.add_argument("--output", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--mlp_ratio", type=float)
    parser.set_defaults(
        model = 'EVA02-CLIP-bigE-14-plus-prune-full',
        pretrained = "./out/eva_clip/eva_clip_2024-08-25_16-15-52/ckpt/student_EVA02-CLIP-bigE-14-plus_10_clip_whole.pth", 
        model_type = "eva_clip",
        task = 'zeroshot_classification',
        batch_size = 512,
        model_cache_dir = None,
        device = 'cuda',
        dataset = 'imagenet1k',
        dataset_root = '/public/scccse/dataset/ILSVRC2012',
        split = 'test',
        annotation_file = '',
        language = 'en',
        cupl = False,
        wds_cache_dir = None,
        num_workers = 4,
        amp = True,
        verbose = True,
        save_clf = None,
        load_clfs = [],
        mlp_ratio=4.0
    )

    args = parser.parse_args()

    return args


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def evaluate(args):
    ddp_setup()
    rank = get_rank()

    dataset_name = args.dataset
    task = args.task

    dataset_root = args.dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace('/', '-'))

    if args.model_type == "eva_clip":
        model, transform, tokenizer = load_clip(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device
        )
        if len(args.load_clfs) != 0:
            model = model.visual
    elif args.model_type == "openclip":
        # openclip_model_and_transform, openclip_tokenizer
        model, _, transform = openclip_model_and_transform(args.model, args.pretrained)
        if len(args.load_clfs)!=0:
            model = model.visual
        tokenizer = openclip_tokenizer(args.model)
        model.cuda()
    elif args.model_type == "internvl":
        model, transform, tokenizer = load_intern_vit(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device,
            mlp_ratio=args.mlp_ratio
        )
        model.to(torch.float16).cuda()
        
    if rank == 0:
        print(transform)

    model.eval()

    dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform,
        split=args.split,
        annotation_file=args.annotation_file,
        download=False,
        language=args.language,
        task=task,
        cupl=args.cupl,
        wds_cache_dir=args.wds_cache_dir,
    )

    collate_fn = get_dataset_collate_fn(args.dataset)

    if args.verbose and rank == 0:
        try:
            print(f'Dataset size: {len(dataset)}')
        except TypeError:
            print('IterableDataset has no len()')
        print(f'Dataset split: {args.split}')
        try:
            print(f'Dataset classes: {dataset.classes}')
            print(f'Dataset number of classes: {len(dataset.classes)}')
        except:
            print('Dataset has no classes.')


    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)

    if args.dataset.startswith('wds/'):
        dataloader = torch.utils.data.DataLoader(
            dataset.batched(args.batch_size), batch_size=None,
            sampler=sampler, num_workers=args.num_workers,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=sampler, num_workers=args.num_workers,
            collate_fn=collate_fn
        )

    if task == 'zeroshot_classification':
        zeroshot_templates = dataset.templates if hasattr(dataset, 'templates') else None
        if args.cupl:
            assert (zeroshot_templates is not None), 'Dataset does not support CuPL prompts'
        # if args.verbose:
        #     print(f"Zero-shot templates: {zeroshot_templates}")
        classnames = dataset.classes if hasattr(dataset, 'classes') else None
        assert (zeroshot_templates is not None and classnames is not None), 'Dataset does not support classification'
        metrics = zeroshot_classification_ddp.evaluate(
            model,
            dataloader,
            tokenizer,
            classnames, zeroshot_templates,
            device=args.device,
            amp=args.amp,
            # verbose=args.verbose,
            verbose=False,
            cupl=args.cupl,
            save_clf=args.save_clf,
            load_clfs=args.load_clfs,
        )

        if rank == 0:
            print(metrics)
            metrics_file_path = os.path.join(Path(args.pretrained).parent, "results_eval_zs.json")
            with open(metrics_file_path, "a") as f:
                f.write(json.dumps({"path": args.pretrained}) + "\n")
                f.write(json.dumps({"dataset": args.dataset}) + "\n")
                for k, v in metrics.items():
                    f.write(json.dumps({k: v}) + "\n")
    destroy_process_group()


if __name__ == '__main__':
    args = get_args()
    evaluate(args)

