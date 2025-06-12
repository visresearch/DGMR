#!/bin/bash  
#SBATCH -o out/prune_eva.%j.out ##作业的输出信息文件  
#SBATCH -J prune_eva ##作业名  
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:4 ##每个作业占用的GPU数量 *



python clip/prune.py

