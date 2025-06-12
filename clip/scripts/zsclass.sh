#!/bin/bash  
#SBATCH -o ./out/zsclass.%j.out ##作业的输出信息文件  
#SBATCH -J zsclass ##作业名  
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *



# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus" \
#   --model_type "eva_clip" \
#   --pretrained "/public/scccse/model_weight/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 64 \
#   --output './out/zsclass-in1k.csv' \
#   --num_workers 8

# "acc1": 0.81994
# "acc5": 0.96624
# "mean_per_class_recall": 0.81998


# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_abs/56-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 256 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune.csv' \
#   --num_workers 8

# "acc1": 0.52648
# "acc5": 0.81432
# "mean_per_class_recall": 0.52698


# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_abs/60-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 512 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune.csv' \
#   --num_workers 8

# "acc1": 0.76608
# "acc5": 0.95048
# "mean_per_class_recall": 0.76614


# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_abs/62-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 384 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_62-63_1792.csv' \
#   --num_workers 8

# "acc1": 0.79896
# "acc5": 0.95974
# "mean_per_class_recall": 0.79882


# -------------------------- scale 0.3 --------------------------
# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_abs/62-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 256 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_62-63_1792_scaled_0.3.csv' \
#   --num_workers 8

# "acc1": 0.78616
# "acc5": 0.95422
# "mean_per_class_recall": 0.78612

# "acc1": 0.78956
# "acc5": 0.95626
# "mean_per_class_recall": 0.78938

# "acc1": 0.77224
# "acc5": 0.95018
# "mean_per_class_recall": 0.77212


# -------------------------- elimination --------------------------
# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_elimination/62-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 256 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_elimination_62-63_1792.csv' \
#   --num_workers 8

# "acc1": 0.8022
# "acc5": 0.96144
# "mean_per_class_recall": 0.8018

# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune-full" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_elimination/0-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 256 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_elimination_0-63_1792.csv' \
#   --num_workers 8


# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_elimination/56-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 256 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_elimination_56-63_1792.csv' \
#   --num_workers 8

# "acc1": 0.43142
# "acc5": 0.7172
# "mean_per_class_recall": 0.43152

# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_elimination/60-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 256 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_elimination_60-63_1792.txt' \
#   --num_workers 8

# "acc1": 0.75888
# "acc5": 0.9472
# "mean_per_class_recall": 0.75882


# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_dim_order/60-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 128 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_dim_order_60-63_1792.txt' \
#   --num_workers 8

# "acc1": 0.76336, "acc5": 0.94974, "mean_per_class_recall": 0.76346


# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva-clip_multi_stage_dim_order/56-63_1792/EVA02_CLIP_E_psz14_plus_s9B_prune.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 128 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_dim_order_56-63_1792.txt' \
#   --num_workers 8

# acc1": 0.50324, "acc5": 0.7906, "mean_per_class_recall": 0.50348

# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune-full" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva_clip/eva_clip_2024-07-30_16-58-58/ckpt/student_EVA02-CLIP-bigE-14-plus-full_5_clip.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 128 \
#   --output './out/zsclass-in1k-EVA02-CLIP-bigE-14-plus-prune_distill_0-63_1792_5.txt' \
#   --num_workers 8

# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus" \
#   --model_type "eva_clip" \
#   --pretrained "/public/scccse/model_weight/EVA-CLIP/EVA02_clip_E_psz14_plus_s9B.pth" \
#   --language "en" \
#   --task "zeroshot_classification" \
#   --dataset "imagenet1k" \
#   --dataset_root /public/scccse/dataset/ILSVRC2012 \
#   --batch_size 128 \
#   --output './out/zsclass-in1k-EVA02_clip_E_psz14_plus_s9B.txt' \
#   --num_workers 2


python clip/clip_benchmark/cli.py eval \
  --model "EVA02-CLIP-bigE-14-plus-prune-full" \
  --model_type "eva_clip" \
  --pretrained "./out/eva_clip/eva_clip_2024-08-01_20-36-28/ckpt/student_EVA02-CLIP-bigE-14-plus-full_5_clip.pth" \
  --language "en" \
  --task "zeroshot_classification" \
  --dataset "imagenet1k" \
  --dataset_root /public/scccse/dataset/ILSVRC2012 \
  --batch_size 128 \
  --output './out/eva_clip/eva_clip_2024-08-01_20-36-28/ckpt/zsclass-in1k-EVA02_clip_E_psz14_plus_s9B.txt' \
  --num_workers 2

python clip/clip_benchmark/cli.py eval \
  --model "EVA02-CLIP-bigE-14-plus-prune" \
  --model_type "eva_clip" \
  --pretrained "./out/eva_clip/eva_clip_2024-08-02_18-58-19/ckpt/student_EVA02-CLIP-bigE-14-plus-full_5_clip.pth" \
  --language "en" \
  --task "zeroshot_classification" \
  --dataset "imagenet1k" \
  --dataset_root /public/scccse/dataset/ILSVRC2012 \
  --batch_size 128 \
  --output './out/eva_clip/eva_clip_2024-08-02_18-58-19/ckpt/zsclass-in1k-EVA02_clip_E_psz14_plus_s9B.txt' \
  --num_workers 2

# "acc1": 0.53166, "acc5": 0.81522, "mean_per_class_recall": 0.53162

python clip/clip_benchmark/cli.py eval \
  --model "EVA02-CLIP-bigE-14-plus-prune" \
  --model_type "eva_clip" \
  --pretrained "./out/eva_clip/eva_clip_2024-08-04_11-25-02/ckpt/student_EVA02-CLIP-bigE-14-plus-full_4_clip.pth" \
  --language "en" \
  --task "zeroshot_classification" \
  --dataset "imagenet1k" \
  --dataset_root /public/scccse/dataset/ILSVRC2012 \
  --batch_size 128 \
  --output './out/eva_clip/eva_clip_2024-08-04_11-25-02/ckpt/zsclass-in1k-EVA02_clip_E_psz14_plus_s9B.txt' \
  --num_workers 2

# "acc1": 0.50402, "acc5": 0.7933, "mean_per_class_recall": 0.5044


python clip/clip_benchmark/cli.py eval \
  --model "EVA02-CLIP-bigE-14-plus-prune" \
  --model_type "eva_clip" \
  --pretrained "./out/eva_clip/eva_clip_2024-08-05_19-35-23/ckpt/student_EVA02-CLIP-bigE-14-plus_5_clip_whole.pth" \
  --language "en" \
  --task "zeroshot_classification" \
  --dataset "imagenet1k" \
  --dataset_root /public/scccse/dataset/ILSVRC2012 \
  --batch_size 128 \
  --output './out/eva_clip/eva_clip_2024-08-05_19-35-23/ckpt/zsclass-in1k-EVA02_clip_E_psz14_plus_s9B.txt' \
  --num_workers 2

# "acc1": 0.49156, "acc5": 0.78944, "mean_per_class_recall": 0.492


python clip/clip_benchmark/cli.py eval \
  --model "EVA02-CLIP-bigE-14-plus-prune" \
  --model_type "eva_clip" \
  --pretrained "./out/eva_clip/eva_clip_2024-08-06_14-09-04/ckpt/student_EVA02-CLIP-bigE-14-plus_1_clip_whole.pth" \
  --language "en" \
  --task "zeroshot_classification" \
  --dataset "imagenet1k" \
  --dataset_root /public/scccse/dataset/ILSVRC2012 \
  --batch_size 128 \
  --output './out/eva_clip/eva_clip_2024-08-06_14-09-04/ckpt/zsclass-in1k-EVA02_clip_E_psz14_plus_s9B.txt' \
  --num_workers 2
# {"acc1": 0.62338, "acc5": 0.8851, "mean_per_class_recall": 0.6232