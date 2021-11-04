#!/bin/bash

#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --job-name=AF_gpu
####SBATCH --comment=dcgmstats
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iskander.j@wehi.edu.au
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -o %x_%j.out

source /stornext/System/data/apps/anaconda3/anaconda3-4.3.1/etc/profile.d/conda.sh

conda activate alphafold_gpu
cd /stornext/HPCScratch/home/iskander.j/git_projects/alphafold2.0


export NVIDIA_VISIBLE_DEVICES=0


./run_alphafold.sh  -d /vast/projects/alphafold/databases \
        -o $2 \
        -m $3 \
        -t 2021-05-30 \
        -i $1 \
        -g true -a 0 \
