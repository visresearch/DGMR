import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import datetime
import time
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, ConcatDataset
from torch.distributed import init_process_group, destroy_process_group, get_world_size, get_rank
import torch.nn.functional as F
import sys

from dinov2.utils.utils import CosineScheduler
from module.module import  build_whole_model_openclip
from module.anyprecision_optimizer import AnyPrecisionAdamW

from utils.logger import console_logger, Logger
from utils.misc import copy_files, AverageMeter

from utils.misc import gather_features
from module.flickr30k_dataset import flickr30k_train
from clip.open_clip import get_tokenizer as openclip_tokenizer
from clip.eva_clip import get_tokenizer as evaclip_tokenizer, create_model_and_transforms as eva_model_and_transform

def ddp_setup():
    init_process_group(backend="nccl")
    # print('LOCAL_RANK:', os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def apply_lr_scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
def apply_wd_scheduler(optimizer, wd):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = wd

def train_one_epoch(args, cfg, model, train_loader, optimizers, schedulers, epoch, loggers):

    model.train()
    optimizer, scaler = optimizers
    lr_scheduler, wd_scheduler = schedulers
    logger_tb, logger_console = loggers
    rank = get_rank()
    world_size = get_world_size()

    niter_per_epoch = len(train_loader)
    train_loader_iter = iter(train_loader)

    times = AverageMeter('Time')
    losses = AverageMeter('Loss')

    niter_global = epoch * niter_per_epoch

    image, text = next(train_loader_iter)

    image = image.bfloat16().cuda(non_blocking=True)
    text = text.cuda(non_blocking=True)

    batch_size= image.shape[0]

    end = time.time()
    end_ = end

    for i in range(niter_per_epoch):
        if i < niter_per_epoch - 1:
            _image, _text = next(train_loader_iter)

            _image = _image.bfloat16().cuda(non_blocking=True)
            _text = _text.cuda(non_blocking=True)

        if cfg.optim.use_lr_scheduler:
            apply_lr_scheduler(optimizer, lr_scheduler[niter_global])
        if cfg.optim.use_wd_scheduler:
            apply_wd_scheduler(optimizer, wd_scheduler[niter_global])

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if args.frame == "internvit" and cfg.model.freeze == "visual":
                with torch.no_grad():
                    text_features = model.module.encode_text(text)
                image_features = model.module.encode_image(image)
            else:
                with torch.no_grad():
                    image_features = model.module.encode_image(image)
                text_features = model.module.encode_text(text)
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                False, True, rank, world_size)

            logits_per_image = model.module.logit_scale.exp() * image_features @ all_text_features.T
            logits_per_text = model.module.logit_scale.exp() * text_features @ all_image_features.T
            labels = torch.arange(batch_size).to(rank) + rank * batch_size
            labels = labels.cuda()
            loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        if i < niter_per_epoch - 1:
            image = _image
            text = _text

        loss_value = loss.item()
        losses.update(loss_value)
        times.update(time.time() - end)
        end = time.time()
        
        if rank == 0 and ((niter_global + 1) % args.print_freq == 0):

            if logger_console is not None and logger_tb is not None:

                logger_tb.add_scalar('loss_iter', loss_value, niter_global)

                lr = optimizer.param_groups[0]['lr']
                logger_tb.add_scalar('lr_iter', lr, niter_global)


                info = f'Epoch [{(epoch + 1):04d}][{(i + 1):04d}/{niter_per_epoch:04d}] - ' \
                    + f'batch_time: {times.val:.2f}s, ' \
                    + f'rest_time: {datetime.timedelta(seconds=int(times.avg * (niter_per_epoch * cfg.optim.nepochs - niter_global)))}, ' \
                    + f'lr: {lr:.2e}, '
                
                info += f'loss: {losses.val:.2e}({losses.avg:.2e})'
                logger_console.info(info)
        
        niter_global += 1
    if logger_console is not None:
        logger_console.info(f'Training Time for 1 epoch: {datetime.timedelta(seconds=time.time() - end_)}')
        
    return losses.avg

def main(args, cfg):
    # --------------------- Env ---------------------
    ddp_setup()
    gpu_id = int(os.environ["LOCAL_RANK"])
    rank = get_rank()
    world_size = get_world_size()
    # --------------------- Log ---------------------
    args.log_dir = f'./out/ft/{cfg.model.arch}'

    if rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

        logger_tb = Logger(args.log_dir, name=args.frame)
        logger_console = console_logger(logger_tb.log_dir, 'console')

        logger_console.info(
            "{}".format(args).replace(', ', ',\n')
        )
    else:
        logger_tb, logger_console = None, None

    transform_train = None
    if args.frame == 'eva_clip':
        model, transform_train, _ = eva_model_and_transform(cfg.model.arch, cfg.model.pretrained, force_custom_clip=True)
        tokenizer = evaclip_tokenizer(cfg.model.arch)
        for p in model.visual.parameters():
            p.requires_grad = False
    elif args.frame == "openclip":
        model, transform_train, _ = build_whole_model_openclip(cfg.model)
        tokenizer = openclip_tokenizer(cfg.model.arch)
        for p in model.visual.parameters():
            p.requires_grad = False
            
    model.to(torch.bfloat16)
    total_num = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"model params: {total_num}B")
    model.cuda()


    train_dataset = flickr30k_train(transform=transform_train,
                    image_root=args.image_root,
                    ann_root=args.ann_root,
                    tokenizer=tokenizer,
                    unique=True
                    )

    train_sampler = DistributedSampler(train_dataset, shuffle=False)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.data.batch_size_per_gpu,
                              shuffle=(train_sampler is None),
                              num_workers=cfg.data.num_workers,
                              sampler=train_sampler,
                              pin_memory=cfg.data.pin_memory,
                              drop_last=True,
                              prefetch_factor=cfg.data.prefetch_factor,
                              persistent_workers=True,
                              )

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu_id], 
        find_unused_parameters=False,
    )
    batch_size = cfg.data.batch_size_per_gpu * world_size
    lr = cfg.optim.base_lr * batch_size / 256
    optimizer = AnyPrecisionAdamW([param for param in model.parameters() if param.requires_grad], lr=lr, betas=(0.9, 0.95),
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        use_kahan_summation=True,
    )
    scaler = None

    optimizer.zero_grad(set_to_none=True)


    if rank == 0:
        save_dir = logger_tb.log_dir
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    niter_epoch = len(train_loader)
    lr_scheduler = CosineScheduler(
        base_value=lr,
        final_value=cfg.optim.min_lr,
        warmup_iters=int(cfg.optim.warmup_epochs * niter_epoch),
        total_iters=int(cfg.optim.nepochs * niter_epoch),
    )

    wd_scheduler = CosineScheduler(
        base_value=cfg.optim.weight_decay,
        final_value=cfg.optim.weight_decay_end,
        total_iters=int(cfg.optim.nepochs * niter_epoch),
    )
    for epoch in range(cfg.optim.nepochs):
        train_one_epoch(args, cfg, model, train_loader, (optimizer, scaler),(lr_scheduler, wd_scheduler), epoch,
                        (logger_tb, logger_console))
        if epoch >= (cfg.optim.nepochs-10) and rank==0:
            fname = cfg.model.arch + f"_{epoch+1}.pth"
            torch.save(model.module.state_dict(), os.path.join(save_dir, fname))

    torch.distributed.barrier()
    destroy_process_group()

    return


if __name__ == '__main__':
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser("Distill for pruned model")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--frame", type=str)
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--ann_root", type=str)

    parser.set_defaults(
        config_file = "configs/eva/clipe/taylor_prune_evaclip.yaml",
        frame = 'eva_clip',
        image_root = '/public/scccse/dataset/Flick30k/Images/',
        ann_root = '/public/scccse/dataset/Flick30k',
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)
    if args.base_lr is not None:
        cfg.optim.base_lr = args.base_lr
    main(args, cfg)

    