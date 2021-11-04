#!/bin/bash
#alphafold preprocessing pipeline submission script, using cpus only
#version 0.1 - 10sep21

#SBATCH --time=48:00:00
#SBATCH --job-name=AF2_cpu
#SBATCH --mem 50G
#SBATCH --cpus-per-task=20
#SBATCH -p regular
#SBATCH --export=ALL
#SBATCH -o logs/%x_%j.out



export ALPHAFOLD_HHBLITS_CPUS=16
export ALPHAFOLD_HHSEARCH_CPUS=8


today=$(date +%F)

module purge
module load alphafold/2.0.1


echo ${outdir},$today,${fastafile}
# Run features first:
alphafold -o ${outdir} -t $today -m model_1 -f ${fastafile}
