**Setup of environment for data preparation and inference**

**Initialize the environment with python 3.10**
conda create --name BAP python=3.10
conda activate graphgen

**Install packages:**
conda install -c conda-forge numpy rdkit
conda install -c huggingface transformers
pip install ankh
conda install biopython
conda install pytorch=2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=*=*cu117 -c pyg

**Optional for training:**
conda install wandb --channel conda-forge
