import os
import argparse

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import json
import torch
from torch.distributed import init_process_group, destroy_process_group, get_rank

from clip.clip_benchmark.datasets.builder import (build_dataset, get_dataset_collate_fn)
from clip.clip_benchmark.models import load_clip, load_clip_visual

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

    parser.set_defaults(
        # model = 'EVA02-CLIP-bigE-14-plus',
        # model = 'EVA02-CLIP-bigE-14-plus-prune',
        model = 'EVA02-CLIP-bigE-14-plus-prune-full',
        # ----------------- teacher -----------------
        # pretrained = "/public/scccse/model_weight/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt",
        # ----------------- svd pruned -----------------
        # 'acc1': 0.52632, 'acc5': 0.81422, 'mean_per_class_recall': 0.52688
        # pretrained = "./out/eva-clip_multi_stage_abs/56-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth",
        # 'acc1': 0.04026, 'acc5': 0.0847, 'mean_per_class_recall': 0.04048
        # pretrained = "./out/eva-clip_multi_round/0-63_7168/EVA02_visual_E_psz14_plus_s9B_prune_clip.pth",
        # 'acc1': 0.66036, 'acc5': 0.90662, 'mean_per_class_recall': 0.66052
        # pretrained = "./out/eva-clip_multi_round/56-63_3584/EVA02_visual_E_psz14_plus_s9B_prune_partial_clip_whole.pth", 
        # 'acc1': 0.81638, 'acc5': 0.96474, 'mean_per_class_recall': 0.81642
        # pretrained = "./out/eva-clip_multi_round/63-63_3584/EVA02_visual_E_psz14_plus_s9B_prune_partial_clip_whole.pth", 
        # 'acc1': 0.81918, 'acc5': 0.96564, 'mean_per_class_recall': 0.81918
        # pretrained = "./out/eva-clip_multi_round/63-63_7168/EVA02_visual_E_psz14_plus_s9B_prune_partial_clip_whole.pth", 
        # 'acc1': 0.7557, 'acc5': 0.9504, 'mean_per_class_recall': 0.75548
        # pretrained = "./out/eva-clip_multi_round/56-63_7168/EVA02_visual_E_psz14_plus_s9B_prune_partial_clip_whole.pth",
        # 'acc1': 0.77586, 'acc5': 0.95394, 'mean_per_class_recall': 0.77578
        # pretrained = "./out/eva-clip_multi_round/60-63_3584/EVA02_visual_E_psz14_plus_s9B_prune_partial_clip_whole.pth", 
        # 'acc1': 0.79456, 'acc5': 0.96028, 'mean_per_class_recall': 0.79486
        # pretrained = "./out/eva-clip_multi_round/60-63_7168/EVA02_visual_E_psz14_plus_s9B_prune_partial_clip_whole.pth", 
        # ----------------- distill -----------------
        pretrained = "./out/eva_clip/eva_clip_2024-08-25_16-15-52/ckpt/student_EVA02-CLIP-bigE-14-plus_10_clip_whole.pth", 
        # pretrained = "./out/eva_clip/[0-63-0.80814]eva_clip_2024-08-16_00-11-27/ckpt/student_EVA02-CLIP-bigE-14-plus_5_clip_whole.pth", 
        model_type = "eva_clip",
        task = 'zeroshot_classification',
        # batch_size = 128,
        batch_size = 512,
        # batch_size = 1, # 3080ti
        model_cache_dir = None,
        device = 'cuda',
        dataset = 'imagenet1k',
        dataset_root = '/public/scccse/dataset/ILSVRC2012',
        # output = './out/eva_clip/zsclass-in1k-EVA02_clip_E_psz14_plus_s9B.txt',
        split = 'test',
        annotation_file = '',
        language = 'en',
        cupl = False,
        wds_cache_dir = None,
        num_workers = 4,
        amp = True,
        verbose = True,
        save_clf = None,
        # save_clf = './out/eva_clip/text_classifier_weight.pth',
        # load_clfs = ['./out/eva_clip/text_classifier_weight+clipe.pth'],
        load_clfs = [],
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

    if len(args.load_clfs) == 0:
        model, transform, tokenizer = load_clip(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device
        )
    else:
        model, transform, tokenizer = load_clip_visual(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device
        )

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

