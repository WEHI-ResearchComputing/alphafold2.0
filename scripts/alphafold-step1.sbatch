#!/bin/bash
#alphafold preprocessing pipeline submission script, using cpus only


#SBATCH --time=48:00:00
#SBATCH --job-name=AF2_pre
#SBATCH --mem 50G
#SBATCH --cpus-per-task=20
#SBATCH -p regular
#SBATCH --export=ALL
#SBATCH -o logs/%x_%j.out



export ALPHAFOLD_HHBLITS_CPUS=16
export ALPHAFOLD_HHSEARCH_CPUS=8


today=$(date +%F)

module purge
module load alphafold/2.0.2


echo ${outdir},$today,${fastafile}
# Run features only
alphafold -o ${outdir} -t $today -m model_1 -f ${fastafile}
