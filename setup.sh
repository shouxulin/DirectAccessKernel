#!/usr/bin/env bash
set -euo pipefail

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O miniconda.sh
bash miniconda.sh -b -f

~/miniconda3/bin/conda init bash
set +u
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda activate base
set -u

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda install -y -c conda-forge python cuda-toolkit=13.0.2 cutlass
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install numpy transformers==5.3.0 accelerate
