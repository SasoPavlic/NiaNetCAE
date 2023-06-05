#!/bin/bash
## Running code on SLURM cluster
##https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html
#SBATCH -J nianet
#SBATCH -o nianet-cae-%j.out
#SBATCH -e nianet-cae-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

#singularity exec -e --pwd /app -B /ceph/grid/home/sasop/Result-long:/app/Result,/ceph/grid/home/sasop/checkpoint-long:/app/checkpoint --nv docker://spartan300/urv-model:1 python ./train.py
singularity exec -e --pwd /app -B $(pwd)/logs:/app/logs,$(pwd)/data:/app/data --nv docker://spartan300/nianet:cae python main.py
