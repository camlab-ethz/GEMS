#!/bin/bash
#SBATCH --job-name=leakage
#SBATCH --output=data_leakage_complexes_c0.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate graphgen

python data_leakage_complexes.py --data_split c0