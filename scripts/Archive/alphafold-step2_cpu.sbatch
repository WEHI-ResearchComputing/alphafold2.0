#!/bin/bash


#SBATCH --time=50:00:00
#SBATCH --job-name=AF2_model_cpu
#SBATCH --mem 100G
#SBATCH --cpus-per-task=10
#SBATCH --export=ALL
#SBATCH -o logs/%x_%j.out
#SBATCH --qos=regular_partitiontimelimit

 
module purge
module load alphafold/2.0.1
echo "Running AlphaFold  $1 inference on CPU"
today=$(date +%F)
alphafold -o ${outdir} -t $today -m $1 ${fastafile}