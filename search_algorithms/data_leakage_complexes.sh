#!/bin/bash
#SBATCH --job-name=SEARCH
#SBATCH --output=search_algos.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate graphgen

python data_leakage_complexes.py --data_split c9