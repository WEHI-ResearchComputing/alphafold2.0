#!/bin/bash

fasta=$(basename -s fasta $1)
if [ -z "$4" ]
then
    maxtime=2020-05-01
maxtime=$2
echo $fasta
j1=$(dsbatch --job-name "m_"$fasta"_c" job_cpu.slurm $1 ./output)
echo "Submitted $j1"
j2=$(dsbatch --dependency=afterok:$j1 --job-name "m_"$fasta"_g" job_gpu.slurm $1 ./output model_1 $maxtime)
echo "Submitted $j2"
j3=$(dsbatch --dependency=afterok:$j1 --job-name "m_"$fasta"_g" job_gpu.slurm $1 ./output model_2 $maxtime)
echo "Submitted $j3"
j4=$(dsbatch --dependency=afterok:$j1 --job-name "m_"$fasta"_g" job_gpu.slurm $1 ./output model_3 $maxtime)
echo "Submitted $j4"
j5=$(dsbatch --dependency=afterok:$j1 --job-name "m_"$fasta"_g" job_gpu.slurm $1 ./output model_4 $maxtime)
echo "Submitted $j5"
j6=$(dsbatch --dependency=afterok:$j1 --job-name "m_"$fasta"_g" job_gpu.slurm $1 ./output model_5 $maxtime)
echo "Submitted $j6"