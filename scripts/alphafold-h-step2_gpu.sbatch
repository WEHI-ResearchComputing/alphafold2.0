#!/bin/bash


#SBATCH --time=48:00:00
#SBATCH --job-name=AF2_gpu
#SBATCH --mem 30G
#SBATCH --cpus-per-task=10
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH -o logs/%x_%j.out

 
module purge
module load alphafold/2.0.2
echo "Running AlphaFold  $1 inference on GPU"
today=$(date +%F)

echo "The cmd is alphafold -o ${outdir} -t $today -m $1 -e $2 ${fastafile} "
alphafold -o ${outdir} -t $today -m $1 -e $2 ${fastafile} 