#!/bin/bash



read -e -p 'Output directory [here]:' outdir
outdir="${outdir:-`pwd -P`}"
outdir=`realpath $outdir`
cd $outdir
echo "Found the following fasta files" 
fastafiles=()
for fastafile in *.fasta; do 
    echo $fastafile
    fastafiles+=($fastafile)
done
## printf '%s\n' "${fastafiles[@]}"

read -p 'Do you wish to run preprocessing pipeline? [y]/n: ' proceed
proceed="${proceed:-"y"}"
read -p 'Do you wish to run model inference pipeline? [y]/n: ' m_inf
m_inf="${m_inf:-"y"}"
if [ $m_inf = "y" ]
then
    read -e -p 'How many models? [1]' num_models        #added to read in a new variable for the number of models
    num_models="${num_models:-1}"                       #set num_models to a default of 1
    models=("model_1" "model_2" "model_3" "model_4" "model_5") ## array of model names
    read -p 'Do you wish to run on gpus? [y]/n: ' gpus
    gpus="${gpus:-"y"}"
    read -e -p 'How many homo-oligomers to use? [1]' num_h        #added to read in a new variable for the number of models
    num_h="${num_h:-1}"   
fi
echo "Output directory : "${outdir} 
#export variable to environment for sbatch
export fastafile outdir 
#create logs folder in output directory
mkdir -p ${outdir}/logs

if [ $proceed = "n" ] && [ $m_inf = "n" ]  ##Nothing to run!! 
then
    echo "Nothing more to do. Bye!!"
    exit 0
elif [ $proceed = "n" ]  ##Run only model inf
then
    for fastafile in "${fastafiles[@]}"
    do
        fasta=`basename -- $fastafile`
        fastaname=${fasta%.*}
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
        if [ $gpus = "y" ]
        then
            j=$(dsbatch --export=fastafile,outdir --dependency=afterok:$j1 --chdir=${outdir} alphafold-h-step2_gpu.sbatch ${models[0]]} $num_h)
        else
            j=$(dsbatch --export=fastafile,outdir --dependency=afterok:$j1 --chdir=${outdir} alphafold-h-step2_cpu.sbatch ${models[0]]} $num_h)
        fi
        
        nmodels=$(( $num_models-1 ))
        if [ $nmodels -ge 1 ]
        then
            for m in $(seq 1 $nmodels)
            do
                if [ $gpus = "y" ]
                then
                    j=$(dsbatch --export=fastafile,outdir --dependency=afterok:$j --chdir=${outdir} alphafold-h-step2_gpu.sbatch ${models[$m]]} $num_h)
                else
                    j=$(dsbatch --export=fastafile,outdir --dependency=afterok:$j --chdir=${outdir} alphafold-h-step2_cpu.sbatch ${models[$m]]} $num_h)
                fi
                echo "Submited model ${models[$m]]} job "$j
            done
        fi
    done
elif [ $proceed = "y" ] && [ $m_inf = "y" ] #Run whole pipeline
then
    j1=0
    for fastafile in "${fastafiles[@]}"
    do
        #launch job
        if [ $j1 = 0 ]
        then
            j1=$(dsbatch --chdir=${outdir} alphafold-step1.sbatch)
        else
            j1=$(dsbatch --chdir=${outdir} --dependency=afterok:$j1 alphafold-step1.sbatch)
        fi
        echo "Submited preprocessing job "$j1
        
        if [ $gpus = "y" ]
        then
            j=$(dsbatch --export=fastafile,outdir --dependency=afterok:$j1 --chdir=${outdir} alphafold-h-step2_gpu.sbatch ${models[0]]} $num_h)
        else
            j=$(dsbatch --export=fastafile,outdir --dependency=afterok:$j1 --chdir=${outdir} alphafold-h-step2_cpu.sbatch ${models[0]]} $num_h)
        fi
        
        nmodels=$(( $num_models-1 ))
        if [ $nmodels -ge 1 ]
        then
            for m in $(seq 1 $nmodels)
            do
                if [ $gpus = "y" ]
                then
                    j=$(dsbatch --export=fastafile,outdir --dependency=afterok:$j --chdir=${outdir} alphafold-h-step2_gpu.sbatch ${models[$m]]} $num_h)
                else
                    j=$(dsbatch --export=fastafile,outdir --dependency=afterok:$j --chdir=${outdir} alphafold-h-step2_cpu.sbatch ${models[$m]]} $num_h)
                fi
                echo "Submited model ${models[$m]]} job "$j
            done
        fi
    done
else #Run preprocessing only
    for fastafile in "${fastafiles[@]}"
    do
        j1=$(dsbatch --export=fastafile,outdir --chdir=${outdir} alphafold-step1.sbatch)
        echo "Submited preprocessing job "$j1
    done
fi



