#!/bin/bash  
#SBATCH -o out/retrieval.%j.out ##作业的输出信息文件  
#SBATCH -J retrieval ##作业名  
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=8


# --------------------------------- original ---------------------------------
# --------- coco
# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus" \
#   --model_type "eva_clip" \
#   --pretrained "/public/scccse/model_weight/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt" \
#   --language "en" \
#   --task "zeroshot_retrieval" \
#   --dataset "mscoco_captions" \
#   --dataset_root /public/scccse/dataset/COCO2014 \
#   --batch_size 128 \
#   --output '/public/scccse/model_weight/EVA-CLIP/zeroshot_retrieval_coco.txt' \
#   --num_workers 2

# "image_retrieval_recall@1": 0.5100359916687012, 
# "image_retrieval_recall@5": 0.7473810315132141, 
# "image_retrieval_recall@10": 0.8267892599105835, 

# "text_retrieval_recall@1": 0.6881999969482422, 
# "text_retrieval_recall@5": 0.8776000142097473, 
# "text_retrieval_recall@10": 0.9305999875068665


# --------- flickr30k
# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus" \
#   --model_type "eva_clip" \
#   --pretrained "/public/scccse/model_weight/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt" \
#   --language "en" \
#   --task "zeroshot_retrieval" \
#   --dataset "flickr30k" \
#   --dataset_root /public/datasets/Flick30k \
#   --batch_size 128 \
#   --output '/public/scccse/model_weight/EVA-CLIP/zeroshot_retrieval_flickr30k.txt' \
#   --num_workers 2

# "image_retrieval_recall@1": 0.7893999814987183, 
# "image_retrieval_recall@5": 0.9441999793052673, 
# "image_retrieval_recall@10": 0.9706000089645386, 

# "text_retrieval_recall@1": 0.9490000009536743, 
# "text_retrieval_recall@5": 0.9929999709129333, 
# "text_retrieval_recall@10": 0.9980000257492065

# --------------------------------- distill ---------------------------------
# --------- coco
# python clip/clip_benchmark/cli.py eval \
#   --model "EVA02-CLIP-bigE-14-plus-prune-full" \
#   --model_type "eva_clip" \
#   --pretrained "./out/eva_clip/[crop0.2_0.81866]eva_clip_2024-08-23_00-26-45/ckpt/student_EVA02-CLIP-bigE-14-plus_10_clip_whole.pth" \
#   --language "en" \
#   --task "zeroshot_retrieval" \
#   --dataset "mscoco_captions" \
#   --dataset_root /public/scccse/dataset/COCO2014 \
#   --batch_size 64 \
#   --output './out/eva_clip/[crop0.2_0.81866]eva_clip_2024-08-23_00-26-45/ckpt/zeroshot_retrieval_coco.txt' \
#   --num_workers 2

# "image_retrieval_recall@1": 0.5117952823638916, 
# "image_retrieval_recall@5": 0.7520591616630554, 
# "image_retrieval_recall@10": 0.8285885453224182, 

# "text_retrieval_recall@1": 0.6740000247955322, 
# "text_retrieval_recall@5": 0.8727999925613403, 
# "text_retrieval_recall@10": 0.9283999800682068


# --------- flickr30k
python clip/clip_benchmark/cli.py eval \
  --model "EVA02-CLIP-bigE-14-plus-prune-full" \
  --model_type "eva_clip" \
  --pretrained "./out/eva_clip/[crop0.2_0.81866]eva_clip_2024-08-23_00-26-45/ckpt/student_EVA02-CLIP-bigE-14-plus_10_clip_whole.pth" \
  --language "en" \
  --task "zeroshot_retrieval" \
  --dataset "flickr30k" \
  --dataset_root /public/datasets/Flick30k \
  --batch_size 128 \
  --output './out/eva_clip/[crop0.2_0.81866]eva_clip_2024-08-23_00-26-45/ckpt/zeroshot_retrieval_flickr30k.txt' \
  --num_workers 2

# "image_retrieval_recall@1": 0.79339998960495, 
# "image_retrieval_recall@5": 0.9491999745368958, 
# "image_retrieval_recall@10": 0.9688000082969666, 

# "text_retrieval_recall@1": 0.9300000071525574, 
# "text_retrieval_recall@5": 0.9929999709129333, 
# "text_retrieval_recall@10": 0.9959999918937683