#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --tmp=20G
#SBATCH --gres=gpumem:20G

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate PYG


python inference.py --model_arch GATE18d --dataset_path dataset_casf2016.pt 