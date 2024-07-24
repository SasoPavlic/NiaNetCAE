#!/bin/bash
## Running code on SLURM cluster
##https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html
#SBATCH -J nianet-cae-pso
#SBATCH -o nianet-cae-pso-%j.out
#SBATCH -e nianet-cae-pso-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=8GB  # memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

singularity exec -e \
    --pwd /app \
    -B $(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs \
    --nv docker://spartan300/nianet:cae \
    python main.py -alg particle_swarm
