#!/bin/bash
#SBATCH --job-name=redundancy
#SBATCH --output=remove_redundancy_output.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate graphgen

python characterize_training_redundancy.py