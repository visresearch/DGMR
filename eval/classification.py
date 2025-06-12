import os

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
from torch.distributed import init_process_group, destroy_process_group, get_world_size, get_rank, all_reduce, ReduceOp

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from timm.utils import accuracy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from timm.models.deit import deit_base_distilled_patch16_384, deit_base_distilled_patch16_224, deit_small_distilled_patch16_224, deit_tiny_distilled_patch16_224

from timm.models.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_base_patch4_window12_384, checkpoint_filter_fn

from utils.logger import MetricLogger
from module.module import prune_model

# from fuse import fuse_model

from parse import get_args_classification

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #   .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_distill(data_loader, model, device, frame):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if frame == "deit":
                output, _ = model(images)
                if len(output)==2:
                    output = (output[0] + output[1]) / 2
            elif frame == "swin":
                output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #   .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main(args, cfg):
    # --------------------- Env ---------------------
    ddp_setup()
    gpu_id = int(os.environ["LOCAL_RANK"])
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device('cuda')

    # --------------------- Dataset ---------------------
    if args.frame == 'deit' or args.frame == 'swin':
        eval_crop_ratio = 0.875
        size = int(cfg.data.input_size / eval_crop_ratio)
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=3),
            transforms.CenterCrop(cfg.data.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])


    dataset = datasets.ImageFolder(
        os.path.join(cfg.data.root, 'val'), transform=transform
    )
    
    
    sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.data.batch_size_per_gpu,
        num_workers=cfg.data.num_workers,
        sampler=sampler,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    # --------------------- Model ---------------------
    if args.frame == 'swin':
        if cfg.model.arch == 'swin_base_patch4_window7_224':
            model = swin_base_patch4_window7_224(pretrained=False)
        elif cfg.model.arch == 'swin_base_patch4_window12_384':
            model = swin_base_patch4_window12_384(pretrained=False)
        elif cfg.model.arch == 'swin_small_patch4_window7_224':
            model = swin_small_patch4_window7_224(pretrained=False)
        elif cfg.model.arch == 'swin_tiny_patch4_window7_224':
            model = swin_tiny_patch4_window7_224(pretrained=False)
        
        if 'block_ids' in cfg.model:
            model = prune_model(model, cfg.model)
        
        state_dict = torch.load(cfg.model.weight, map_location='cpu')

        if 'block_ids' not in cfg.model:
            state_dict = checkpoint_filter_fn(state_dict, model)

        incompatible_keys = model.load_state_dict(state_dict, strict=True)
        print(f'Model ({rank}):', incompatible_keys)
    
    model.cuda()

    # --------------------- Evaluation ---------------------
    test_stats = evaluate(data_loader, model, device)

    if rank == 0:
        print(f"Acc@1: {test_stats['acc1']:.2f}, Acc@5: {test_stats['acc5']:.2f}")


if __name__ == '__main__':
    from omegaconf import OmegaConf
    args = get_args_classification()
    cfg = OmegaConf.load(args.config_file)
    main(args, cfg)
