#!/bin/bash

# bash ./create_and_configure_env.sh <env_name>
if [ -z "$1" ]; then
   printf "\nError! Missing the conda environment name.\n\n"
   exit 1
fi
# Create a conda environment
conda create -n "$1" python=3.10.16
source activate base
conda activate "$1"
# Install packages
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install tabulate==0.9.0 tqdm==4.67.1
printf "\nThe environment is completed.\n\nNow type conda activate $1.\n\n"
