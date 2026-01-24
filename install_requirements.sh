#!/bin/bash

# Install packages
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install tabulate==0.9.0 tqdm==4.67.1
