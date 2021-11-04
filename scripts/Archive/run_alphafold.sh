#!/bin/bash


#get name of folder. -e enables tab completion
read -e -p 'Fasta file: ' fastafile
fastafile=`realpath $fastafile`
fasta=`basename -- $fastafile`
fastaname=${fasta%.*}
if [ ! -s $fastafile ]
then
    echo "Fasta file does not exist or has a zerobyte size."
    exit 1
fi
read -e -p 'Output directory [here]:' outdir
outdir="${outdir:-`pwd -P`}"
outdir=`realpath $outdir`
read -e -p 'How many models? [1]' num_models        #added to read in a new variable for the number of models
num_models="${num_models:-1}"                       #set num_models to a default of 1
#model_string=$(echo model_$(seq 1 $num_models))     #creates a string: model_1 2 3 ..$num_models
#model_string_formatted=${model_string// /,model_}   #reformats the model_string so it is the correct convention for alphafold: model_1,model_2,..model_$num_models
models=("model_1" "model_2" "model_3" "model_4" "model_5")
echo "input file: "${fastafile}
echo "Output directory : "${outdir} 

read -p 'Do you wish to run preprocessing pipeline? [y]/n: ' proceed
proceed="${proceed:-"y"}"


#export variable to environment for sbatch
export fastafile outdir 

if [ $proceed = "n" ]
then
    if [ ! -d $outdir/$fastaname ] 
    then
        echo "Output directory ($outdir/$fastaname ) does not exist."
        exit 1
    elif [ ! -d $outdir/$fastaname/msas ] 
    then
        echo "Output msas directory ($outdir/$fastaname/msas) does not exist. You must run preprocessing pipeline first."
        exit 1
    fi
    #launch job
    num_models=$(( $num_models-1 ))
    for m in $(seq 0 $num_models)
    do
      j=$(dsbatch --chdir=${outdir} alphafold-step2.sbatch ${models[$m]]})
      echo "Submited model ${models[$m]]} job "$j
    done
else
    #launch job
    j1=$(dsbatch --chdir=${outdir} alphafold-step1.sbatch)
    echo "Submited preprocessing job "$j1
    num_models=$(( $num_models-1 ))
    for m in $(seq 0 $num_models)
    do
      j=$(dsbatch --dependency=afterok:$j1 --chdir=${outdir} alphafold-step2.sbatch ${models[$m]]})
      echo "Submited model ${models[$m]]} job "$j
    done
fi