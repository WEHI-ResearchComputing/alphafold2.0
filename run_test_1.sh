#!/bin/bash
# Description: AlphaFold non-docker version
# Author: Sanjay Kumar Srikakulam

usage() {
        echo ""
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-d <data_dir>     Path to directory of supporting data"
        echo "-o <output_dir>   Path to a directory that will store the results."
        echo "-m <model_names>  Names of models to use (a comma separated list)"
        echo "-f <fasta_path>   Path to a FASTA file containing one sequence"
        echo "-t <max_template_date> Maximum template release date to consider (ISO-8601 format - i.e. YYYY-MM-DD). Important if folding historical test sets"
        echo "Optional Parameters:"
        echo "-b <benchmark>    Run multiple JAX model evaluations to obtain a timing that excludes the compilation time, which should be more indicative of the time required for inferencing many
    proteins (default: 'False')"
        echo "-g <use_gpu>      Enable NVIDIA runtime to run with GPUs (default: 'True')"
        echo "-a <gpu_devices>  Comma separated list of devices to pass to 'CUDA_VISIBLE_DEVICES' (default: 'all')"
        echo "-p <preset>       Choose preset model configuration - no ensembling (full_dbs) or 8 model ensemblings (casp14) (default: 'full_dbs')"
        echo ""
        exit 1
}

while getopts ":d:o:m:f:t:a:p:b:g:" i; do
        case "${i}" in
        d)
                data_dir=$OPTARG
                
        ;;
        o)
                output_dir=$OPTARG
        ;;
        m)
                model_names=$OPTARG
        ;;
        f)
                fasta_path=$OPTARG
        ;;
        t)
                max_template_date=$OPTARG
        ;;
        b)
                benchmark=$OPTARG
        ;;
        g)
                use_gpu=$OPTARG
                echo "g:"$OPTARG
        ;;
        a)
                gpu_devices=$OPTARG
        ;;
        p)
                preset=$OPTARG
        ;;
        esac
done
