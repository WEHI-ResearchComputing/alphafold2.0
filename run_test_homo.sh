#!/bin/bash

echo "Load Conda Env"
source /stornext/System/data/apps/anaconda3/anaconda3-4.3.1/etc/profile.d/conda.sh
conda activate alphafold_gpu

echo "Running"
export ALPHAFOLD_JACKHMMER_CPUS=20
export ALPHAFOLD_HHBLITS_CPUS=16
export ALPHAFOLD_HHSEARCH_CPUS=8
export ALPHAFOLD_HMMBUILD_CPUS=8
python run_test_homo.py