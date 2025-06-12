import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import datetime
import time
import math
# from omegaconf import OmegaConf
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, ConcatDataset
from torch.distributed import init_process_group, destroy_process_group, get_world_size, get_rank, all_reduce, ReduceOp
from torch.nn.functional import mse_loss, cosine_similarity, cross_entropy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
from torch.optim.swa_utils import AveragedModel
from utils.loss import Knowledge_Distillation_Loss

from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import torchvision.datasets as datasets

from dinov2.utils.utils import CosineScheduler
import torchvision.transforms as transforms_torch
from transforms import get_transforms_dinov2, get_transforms_eva, Mixup, get_transforms_swin, get_transforms_internVIT
from parse import get_args
from module.module import build_model_dinov2, build_model_eva_clip, prune_model, \
    build_model_internVIT, build_model_openclip
from module.anyprecision_optimizer import AnyPrecisionAdamW

from utils.logger import console_logger, Logger
from utils.misc import copy_files, AverageMeter

import dinov2
from dinov2.models.vision_transformer import DinoVisionTransformer

from clip.eva_clip import eva_vit_model
from clip.eva_clip.eva_vit_model import EVAVisionTransformer
from module.model_wrapper import ModelWrapper

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_base_patch4_window12_384, checkpoint_filter_fn
from utils.model_ema import ModelEma
from eval.classification import evaluate_distill as evaluate


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


def train_one_epoch(args, cfg, student, teacher, \
        train_loader, optimizer, scaler, schedulers, epoch, loggers, student_ema):

    student.train()
    teacher.eval()

    rank = get_rank()

    lr_scheduler, wd_scheduler, xent_coef_scheduler = schedulers
    logger_tb, logger_console = loggers

    niter_per_epoch = len(train_loader)
    train_loader_iter = iter(train_loader)

    autocast = nullcontext \
        if cfg.optim.pure_bf16 or cfg.optim.float else torch.cuda.amp.autocast

    times = AverageMeter('Time')
    losses = AverageMeter('Loss')

    niter_global = epoch * niter_per_epoch

    data, labels = next(train_loader_iter)

    data = data.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)

    if cfg.optim.pure_bf16:
        data = data.bfloat16().cuda(non_blocking=True)
    elif cfg.optim.float:
        data = data.float().cuda(non_blocking=True)
    else:
        data = data.half().cuda(non_blocking=True)
        # data = data.cuda(non_blocking=True)

    end = time.time()
    end_ = end

    loss_feat, loss_dist = None, None
    loss_xent, loss_cls = None, None
    if (args.frame == "swin") and cfg.optim.loss_cls == "kl":
        kl_loss = Knowledge_Distillation_Loss(cfg.optim.distill_t)
    for i in range(niter_per_epoch):
        loss = 0
        if i < niter_per_epoch - 1:
            _data, _labels = next(train_loader_iter)
            _labels = _labels.cuda(non_blocking=True)
            if cfg.optim.pure_bf16:
                _data = _data.bfloat16().cuda(non_blocking=True)
            elif cfg.optim.float:
                _data = _data.float().cuda(non_blocking=True)
            else:
                _data = _data.half().cuda(non_blocking=True)
        
        if cfg.optim.use_lr_scheduler:
            apply_lr_scheduler(optimizer, lr_scheduler[niter_global])
        
        if cfg.optim.use_wd_scheduler:
            apply_wd_scheduler(optimizer, wd_scheduler[niter_global])
        
        if xent_coef_scheduler is not None:
            cfg.optim.xent_coef = float(xent_coef_scheduler[niter_global])

        ################## teacher ########################
        with torch.no_grad():
            target, target_feat = teacher(data)

        with autocast():
            pred, feat = student(data)
            if args.frame == 'swin':
                loss_cls = mse_loss(pred, target)
                loss_feat = mse_loss(feat, target_feat)
                if 'xent_coef' in cfg.optim:
                    loss_xent = cross_entropy(pred, labels)
            elif args.frame == 'eva_clip' or args.frame == 'internVIT' or args.frame == "openclip":
                loss_cls = mse_loss(pred, target)
                loss_feat = mse_loss(feat, target_feat)


            if loss_cls is not None:
                if "cls_coef" not in cfg.optim:
                    cfg.optim.cls_coef = 1.0
                loss += cfg.optim.cls_coef * loss_cls
            if loss_dist is not None:
                loss += cfg.optim.dist_coef * loss_dist
            if loss_feat is not None:
                if "feat_coef" not in cfg.optim:
                    cfg.optim.feat_coef = 1.0
                loss += cfg.optim.feat_coef * loss_feat
            if loss_xent is not None:
                loss += cfg.optim.xent_coef * loss_xent
        if cfg.optim.pure_bf16:
            (loss/cfg.optim.step_interval).backward()
        else:
            scaler.scale(loss/cfg.optim.step_interval).backward()
        if (niter_global+1) % cfg.optim.step_interval == 0:
            if cfg.optim.pure_bf16:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if student_ema is not None:
                if isinstance(student, FSDP):                    
                    student_ema.update(student)
                else:
                    student_ema.update(student.module.model)
        torch.cuda.synchronize()

        if i < niter_per_epoch - 1:
            data = _data
            labels = _labels

        loss_value = loss.item()
        losses.update(loss_value)
        times.update(time.time() - end)
        end = time.time()
        
        if rank == 0 and ((niter_global + 1) % (args.print_freq * cfg.optim.step_interval) == 0):

            if logger_console is not None and logger_tb is not None:

                logger_tb.add_scalar('loss_iter', loss_value, niter_global)

                lr = optimizer.param_groups[0]['lr']
                logger_tb.add_scalar('lr_iter', lr, niter_global)

                info = f'Epoch [{(epoch + 1):04d}][{(i + 1):04d}/{niter_per_epoch:04d}] - ' \
                    + f'batch_time: {times.val:.2f}s, ' \
                    + f'rest_time: {datetime.timedelta(seconds=int(times.avg * (niter_per_epoch * cfg.optim.nepochs - niter_global)))}, ' \
                    + f'lr: {lr:.2e}, '
                
                if loss_cls is not None:
                    loss_cls_val = loss_cls.item()
                    info += f'loss_cls: {loss_cls_val:.2e}, '
                    logger_tb.add_scalar('loss_cls', loss_cls_val, niter_global)
                
                if loss_feat is not None:
                    loss_feat_val = loss_feat.item()
                    info += f'loss_feat: {loss_feat_val:.2e}, '
                    logger_tb.add_scalar('loss_feat', loss_feat_val, niter_global)
                
                if loss_dist is not None:
                    loss_dist_val = loss_dist.item()/cfg.optim.step_interval
                    info += f'loss_dist: {loss_dist_val:.2e}, '
                    logger_tb.add_scalar('loss_dist', loss_dist_val, niter_global)

                if loss_xent is not None:
                    loss_xent_val = loss_xent.item()
                    info += f'loss_xent: {loss_xent_val:.2e}, '
                    logger_tb.add_scalar('loss_xent', loss_xent_val, niter_global)
                
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
    if args.frame == 'eva_clip':
        args.log_dir = f'./out/{cfg.model.vision_cfg.arch}'
    elif args.frame == "interVIT":
        args.log_dir = f'./out/{cfg.teacher.model.type}'
    else:
        args.log_dir = f'./out/{cfg.teacher.arch}'

    if rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

        logger_tb = Logger(args.log_dir, name=args.frame)
        logger_console = console_logger(logger_tb.log_dir, 'console')
        dst_dir = os.path.join(logger_tb.log_dir, 'code/')
        copy_files('./', dst_dir, args.exclude_file_list)

        logger_console.info(
            "{}".format(args).replace(', ', ',\n')
        )
    else:
        logger_tb, logger_console = None, None

    # --------------------- Dataset ---------------------
    transform_train = None
    if args.frame == 'dinov2':
        transform_train = get_transforms_dinov2(cfg.data.input_size, cfg.data.min_crop)
    elif args.frame == 'eva_clip':
        teacher, transform_train, _ = build_model_eva_clip(cfg.teacher)
        student, _, _ = build_model_eva_clip(cfg.student)
        state_dict_t = torch.load(cfg.teacher.pretrained, map_location="cpu")
    elif args.frame == 'swin':
        transform_train = get_transforms_swin(cfg.data.input_size, cfg.data.min_crop)
    elif args.frame == "internVIT":
        transform_train = get_transforms_internVIT(cfg, is_train=True)
    elif args.frame == "openclip":
        teacher, transform_train, _ = build_model_openclip(cfg.teacher)
        student, _, _ = build_model_openclip(cfg.student)
        state_dict_t = torch.load(cfg.teacher.pretrained, map_location="cpu")
    
    train_dataset = datasets.ImageFolder(
        os.path.join(cfg.data.root, 'train'), transform=transform_train
    )
    
    if rank == 0:
        logger_console.info(f'Training dataset size: {len(train_dataset)}')

    train_sampler = DistributedSampler(train_dataset)

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

    if args.frame=="swin":
        eval_crop_ratio = 0.875
        size = int(cfg.data.input_size / eval_crop_ratio)
        val_transform = transforms_torch.Compose([
            transforms_torch.Resize(size, interpolation=3),
            transforms_torch.CenterCrop(cfg.data.input_size),
            transforms_torch.ToTensor(),
            transforms_torch.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        val_dataset = datasets.ImageFolder(
            os.path.join(cfg.data.root, 'val'), transform=val_transform
        )
        val_sampler = torch.utils.data.DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=cfg.data.batch_size_per_gpu,
            num_workers=cfg.data.num_workers,
            sampler=val_sampler,
            pin_memory=cfg.data.pin_memory,
            drop_last=False,
            prefetch_factor=cfg.data.prefetch_factor,
        )
    # --------------------- Model ---------------------
    
    if args.frame == 'swin':
        if cfg.teacher.arch == 'swin_base_patch4_window12_384':
            teacher = swin_base_patch4_window12_384(pretrained=False)
            student = prune_model(swin_base_patch4_window12_384(pretrained=False), cfg.student)
        elif cfg.teacher.arch == 'swin_small_patch4_window7_224':
            teacher = swin_small_patch4_window7_224(pretrained=False)
            student = prune_model(swin_small_patch4_window7_224(pretrained=False), cfg.student)
        elif cfg.teacher.arch == 'swin_tiny_patch4_window7_224':
            teacher = swin_tiny_patch4_window7_224(pretrained=False)
            student = prune_model(swin_tiny_patch4_window7_224(pretrained=False), cfg.student)

        if cfg.optim.pure_bf16:
            teacher.to(torch.bfloat16)
        elif cfg.optim.float:
            teacher.to(torch.float)
        else:
            teacher.half()
        
        if cfg.optim.pure_bf16:
            student.to(torch.bfloat16)
        elif cfg.optim.float:
            student.to(torch.float)

        state_dict_t = torch.load(cfg.teacher.weight, map_location='cpu')
       
        incompatible_keys = teacher.load_state_dict(state_dict_t, strict=True)
        print(f'Teacher ({rank}):', incompatible_keys)

        state_dict_s = torch.load(cfg.student.weight, map_location='cpu')
        incompatible_keys = student.load_state_dict(state_dict_s, strict=True)
        print(f'Student ({rank}):', incompatible_keys)
    elif args.frame == 'eva_clip':
        teacher = build_model_eva_clip(cfg.teacher)
        if cfg.optim.pure_bf16:
            teacher.to(torch.bfloat16)
        elif cfg.optim.float:
            teacher.to(torch.float)
        else:
            teacher.half()

        # state_dict_t = teacher.state_dict()

        student = build_model_eva_clip(cfg.student)
        if cfg.optim.pure_bf16:
            student.to(torch.bfloat16)
        elif cfg.optim.float:
            student.to(torch.float)
    elif args.frame == "internVIT":
        teacher = build_model_internVIT(cfg.teacher)
        if cfg.optim.pure_bf16:
            teacher.to(torch.bfloat16)
        elif cfg.optim.float:
            teacher.to(torch.float)
        else:
            teacher.half()
        teacher.cuda()
        student = build_model_internVIT(cfg.student)
        if cfg.optim.pure_bf16:
            student.to(torch.bfloat16)
        elif cfg.optim.float:
            student.to(torch.float)
    elif args.frame == "openclip":
        if cfg.optim.pure_bf16:
            teacher.to(torch.bfloat16)
        elif cfg.optim.float:
            teacher.to(torch.float)
        else:
            teacher.half()
        if cfg.optim.pure_bf16:
            student.to(torch.bfloat16)
        elif cfg.optim.float:
            student.to(torch.float)
    else:
        RuntimeError(f'Not recoginize {args.frame}')
    print("model is done!")
    # --------------------- Resume ---------------------
    max_acc = 0
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        student.load_state_dict(resume_ckpt, strict=True)
        student.cuda()
        if cfg.optim.pure_bf16:
            student.to(torch.bfloat16)
        elif cfg.optim.float:
            student.to(torch.float)
        if args.frame == "swin":
            test_stats = evaluate(val_data_loader, student, device=torch.device(f"cuda:{rank}"), frame=args.frame)
            if test_stats['acc1'] > max_acc:
                max_acc = test_stats['acc1']
            if rank==0:
                print(f"Acc@1: {test_stats['acc1']:.2f}, Acc@5: {test_stats['acc5']:.2f}")
                print(f"maxAcc@1: {max_acc}")
    if args.frame == 'eva_clip':
        state_dict_norm = teacher.norm.state_dict()
        state_dict_head = teacher.head.state_dict()

        teacher.norm = torch.nn.Identity()
        student.norm = torch.nn.Identity()

        teacher.head = torch.nn.Identity()
        student.head = torch.nn.Identity()

    if args.frame == "internVIT":
    
        teacher.norm = torch.nn.Identity()
        student.norm = torch.nn.Identity()
        if cfg.optim.skip_clip_proj:
            teacher.clip_projector = torch.nn.Identity()
            student.clip_projector = torch.nn.Identity()
    
    if args.frame == "openclip":
        if cfg.optim.skip_proj:
            teacher.proj = None
            student.proj = None

    # --------------------- ModelEMA ---------------------
    if 'distribute_strategy' in cfg.optim and cfg.optim.distribute_strategy == 'fsdp':
        cfg.optim.use_ema = False

    if cfg.optim.use_ema:
        student_ema = ModelEma(student, decay=cfg.optim.ema_decay)
        student_ema.ema.float()
    else:
        student_ema = None
    
    if student_ema is not None:
        student_ema.ema.cuda()

    teacher.cuda()
    student.cuda()

    teacher = ModelWrapper(teacher)
    student = ModelWrapper(student)


    if 'distribute_strategy' in cfg.optim and cfg.optim.distribute_strategy == 'fsdp':
        if args.frame == "eva_clip":
            wrapping_policy = partial(transformer_auto_wrap_policy, 
                                        transformer_layer_cls={
                                            eva_vit_model.Block
                                        }
                                    )

        precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )

        student = FSDP(student, 
            auto_wrap_policy=wrapping_policy, 
            mixed_precision=precision, 
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, 
            device_id=gpu_id, 
            limit_all_gathers=True
        )
    else:
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[gpu_id], 
            find_unused_parameters=False,
        )

    # --------------------- Optimizer ---------------------
    batch_size = cfg.data.batch_size_per_gpu * world_size
    # lr = cfg.optim.base_lr * math.sqrt(batch_size / 1024)
    lr = cfg.optim.base_lr * batch_size / 256

    params = student.parameters()

    if cfg.optim.pure_bf16:
        optimizer = AnyPrecisionAdamW(params, lr=lr, betas=(0.9, 0.95),
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=True,
        )
        scaler = None
    else:
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))
        scaler = torch.cuda.amp.GradScaler()

    optimizer.zero_grad(set_to_none=True)
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
    
    if "xent_coef_scheduler" in cfg.optim and cfg.optim.xent_coef_scheduler:
        xent_coef_scheduler = CosineScheduler(
            base_value=0,
            final_value=cfg.optim.xent_coef,
            total_iters=int(cfg.optim.nepochs * niter_epoch),
        )
    else:
        xent_coef_scheduler = None
        

    if rank == 0:
        save_dir = os.path.join(logger_tb.log_dir, 'ckpt')
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()
   
    
    for epoch in range(cfg.optim.nepochs):

        train_loader.sampler.set_epoch(epoch)

        loss_epoch = train_one_epoch(args, cfg, student, teacher, train_loader, \
            optimizer, scaler, (lr_scheduler, wd_scheduler, xent_coef_scheduler), epoch, \
                (logger_tb, logger_console), student_ema)

        
        if isinstance(student, FSDP):
            with FSDP.state_dict_type(
                student, StateDictType.FULL_STATE_DICT, 
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                # state_dict = student.model.state_dict()
                state_dict = student.state_dict()
                nremove = len('model.')
                keys = list(state_dict.keys())
                for k in keys:
                    state_dict[k[nremove:]] = state_dict[k]
                    del state_dict[k]
        else:
            if isinstance(student.module, ModelWrapper):
                state_dict = student.module.model.state_dict()
            else:
                state_dict = student.module.state_dict()
        if student_ema is not None:
            state_dict_ema = student_ema.ema.state_dict()
            
        if ((epoch + 1) % args.save_freq == 0 or epoch + 1 == cfg.optim.nepochs) and rank==0:
            
            fname = f'student_{cfg.student.arch}_{epoch + 1}.pth' \
                if args.frame != 'eva_clip' \
                else f'student_{cfg.model.vision_cfg.arch}_{epoch + 1}.pth'
            
            if args.frame == 'eva_clip':
                state_dict['head.weight'] = state_dict_head['weight']
                if 'bias' in state_dict_head:
                    state_dict['head.bias'] = state_dict_head['bias']
                state_dict['norm.weight'] = state_dict_norm['weight']
                if 'bias' in state_dict_norm:
                    state_dict['norm.bias'] = state_dict_norm['bias']
                
                if student_ema is not None:
                    state_dict_ema['head.weight'] = state_dict_head['weight']
                    if 'bias' in state_dict_head:
                        state_dict_ema['head.bias'] = state_dict_head['bias']
                    state_dict_ema['norm.weight'] = state_dict_norm['weight']
                    if 'bias' in state_dict_norm:
                        state_dict_ema['norm.bias'] = state_dict_norm['bias']
            if args.frame == "openclip":
                for key in state_dict.keys():
                    state_dict_t["visual." + key] = state_dict[key]
                state_dict = state_dict_t
            if args.frame == "internVIT":
                state_dict_t = torch.load(cfg.teacher.model.intern_vit_6b.pretrained, map_location="cpu")
                for key in state_dict.keys():
                    state_dict_t[key] = state_dict[key]
                state_dict = state_dict_t

            torch.save(state_dict, os.path.join(save_dir, fname))
            if student_ema is not None:
                if args.frame == "openclip":
                    for key in state_dict_ema.keys():
                        state_dict_t["visual." + key] = state_dict_ema[key]
                    state_dict_ema = state_dict_t
                torch.save(state_dict_ema, os.path.join(save_dir, fname.replace('.pth', '_ema.pth')))
        

        if args.frame == "swin":
            if isinstance(student.module, ModelWrapper):       
                test_stats = evaluate(val_data_loader, student.module.model, device=torch.device(f"cuda:{rank}"), frame=args.frame)
            else:
                test_stats = evaluate(val_data_loader, student.module, device=torch.device(f"cuda:{rank}"), frame=args.frame)
            if test_stats['acc1']>max_acc:
                max_acc = test_stats['acc1']
                if rank==0:
                    torch.save(state_dict, os.path.join(save_dir,f"{max_acc:.2f}_best_student.pth"))
            if rank==0:
                print(f"Acc@1: {test_stats['acc1']:.2f}, Acc@5: {test_stats['acc5']:.2f}")
                print(f"maxAcc@1: {max_acc}")

            test_stats_ema = evaluate(val_data_loader, student_ema.ema, device=torch.device(f"cuda:{rank}"), frame=args.frame)
            if test_stats_ema['acc1']>max_acc:
                max_acc = test_stats_ema['acc1']
                if rank==0:
                    torch.save(state_dict_ema, os.path.join(save_dir,f"{max_acc:.2f}_best_student_ema.pth"))
            if rank==0:
                print(f"EMA Acc@1: {test_stats_ema['acc1']:.2f}, Acc@5: {test_stats_ema['acc5']:.2f}")
                print(f"maxAcc@1: {max_acc}")
            
            if rank==0:
                logger_tb.add_scalar('acc1_model', test_stats['acc1'], epoch+1)
                logger_tb.add_scalar("acc1_model_ema", test_stats_ema["acc1"], epoch+1)

        if rank == 0:
            lr_ = optimizer.param_groups[0]['lr']

            if logger_tb is not None:
                logger_tb.add_scalar('train_loss_epoch', loss_epoch, epoch+1)
                logger_tb.add_scalar('lr_epoch', lr_, epoch+1)
                logger_tb.flush()
            
            if logger_console is not None:
                logger_console.info(f"Epoch: {epoch + 1}, Lr: {lr_}, Loss: {loss_epoch}")


    if rank == 0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger_console.info(f'Training time: {total_time_str}')

    torch.distributed.barrier()
    destroy_process_group()

    return


if __name__ == '__main__':
    from omegaconf import OmegaConf
    args = get_args()
    cfg = OmegaConf.load(args.config_file)
    if args.base_lr is not None:
        cfg.optim.base_lr = args.base_lr
    main(args, cfg)

