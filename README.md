# AlphaFold in WEHI Milton

For WEHI Milton users, you can run Alphafold from the modules system. Documentation [here](https://rc.wehi.edu.au/Documentation/advanced-tools/alphafold)

```
module load alphafold/2.0.1
```
This repo is a work in progress for experimenting extra features to add to the pipeline.
This repo is based on a modified clone from [DeepMind Alphafold](https://github.com/deepmind/alphafold) and also using the non-docker solutions from 
* https://github.com/kuixu/alphafold
* https://github.com/kalininalab/alphafold_non_docker

This package provides an implementation of the inference pipeline of AlphaFold v2.0. based on [DeepMind Alphafold](https://github.com/deepmind/alphafold). Referenced in this Nature paper
[AlphaFold paper](https://doi.org/10.1038/s41586-021-03819-2). Please also refer
to the [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf)
for a detailed description of the method.

### Genetic databases

The generic databases has been downloaded to `/vast/projects/alphafold/databases/`

### Model parameters

The AlphaFold parameters are made available for non-commercial use only under the terms of the
CC BY-NC 4.0 license. 

The AlphaFold parameters are also available in  available `/vast/projects/alphafold/databases/params` from
https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar:
*   5 models which were used during CASP14, and were extensively validated for
    structure prediction quality (see Jumper et al. 2021, Suppl. Methods 1.12
    for details).
*   5 pTM models, which were fine-tuned to produce pTM (predicted TM-score) and
    predicted aligned error values alongside their structure predictions (see
    Jumper et al. 2021, Suppl. Methods 1.9.7 for details).

## Setting up environment on Milton
```
ssh vc7-shared
```
### Do the following if and only if, first time to try anaconda
```
module load anaconda3

conda init
```

### Installing  env , change $pathtoinstall to a directory on your HPCScratch
```
cd $pathtoinstall
git clone https://github.com/WEHI-ResearchComputing/alphafold.git
cd alphafold
conda create --name alphafold python==3.8
conda activate alphafold
conda install -c conda-forge -y openmm=7.5.1 pdbfixer pip
conda install -y -c bioconda hmmer hhsuite==3.3.0 kalign2

conda install -y -c nvidia cudnn==8.0.4

pip3 install --upgrade pip
pip3 install --user -r ./requirements.txt
pip3 install --upgrade "jax[cuda102]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

work_path=$(pwd)
a=$(which python)
cd $(dirname $(dirname $a))/lib/python3.8/site-packages
patch -p0 < $work_path/docker/openmm.patch
cd $work_path
```

## Citing  AlphaFold 2

```bibtex
@Article{AlphaFold2021,
  author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  journal = {Nature},
  title   = {Highly accurate protein structure prediction with {AlphaFold}},
  year    = {2021},
  doi     = {10.1038/s41586-021-03819-2},
  note    = {(Accelerated article preview)},
}
```


