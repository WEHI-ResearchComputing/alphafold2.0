#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --mem 100G
#SBATCH --cpus-per-task=20
#SBATCH -p regular
#SBATCH --export=ALL
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err


export ALPHAFOLD_HHBLITS_CPUS=16
export ALPHAFOLD_HHSEARCH_CPUS=8


module purge
module load alphafold/2.0.1

# Run features first:
alphafold -o $2 -t $4 -m model_1 -f $1 
