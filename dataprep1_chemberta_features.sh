#!/bin/bash
#SBATCH --job-name=chemberta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --tmp=20G
#SBATCH --gres=gpumem:20G

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate BAP


#python dataprep1_chemberta_features.py --data_dir inference_test --model ChemBERTa-77M-MLM
python -m dataprep.chemberta_features --data_dir inference_test --model ChemBERTa-77M-MLM