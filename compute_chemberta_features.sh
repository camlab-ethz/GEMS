#!/bin/bash
#SBATCH --job-name=ankh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --tmp=20G
#SBATCH --gres=gpumem:20G

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate graphgen


python dataprep1_chemberta_features.py --data_dir /cluster/work/math/dagraber/DTI/PDBbind2 --model ChemBERTa-10M-MLM