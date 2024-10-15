**Setup of environment for data preparation and inference**<br />
<br />
**Initialize the environment with python 3.10**<br />
conda create --name BAP python=3.10<br />
conda activate graphgen<br />
<br />
**Install packages:**<br />
conda install -c conda-forge numpy rdkit <br />
conda install -c huggingface transformers <br />
pip install ankh <br />
conda install biopython <br />
conda install pytorch=2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia <br />
conda install pyg=*=*cu117 -c pyg <br />

**Optional for training:**<br />
conda install wandb --channel conda-forge<br />
