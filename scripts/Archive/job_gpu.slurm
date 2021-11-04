#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem 50G
#SBATCH --cpus-per-task=5
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err


module purge
module load alphafold/2.0.1
alphafold -o $2 -t $4 -m $3  $1