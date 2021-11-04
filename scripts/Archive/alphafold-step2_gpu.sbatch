#!/bin/bash


#SBATCH --time=50:00:00
#SBATCH --job-name=AF2
#SBATCH --mem 30G
#SBATCH --cpus-per-task=10
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH -o logs/%x_%j.out
#SBATCH --qos=regular_partitiontimelimit
 
module purge
module load alphafold/2.0.1

echo "Running AlphaFold  $1 inference on GPU"
today=$(date +%F)
alphafold -o ${outdir} -t $today -m $1 ${fastafile}
