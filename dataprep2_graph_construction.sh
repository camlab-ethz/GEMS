#!/bin/bash
#SBATCH --job-name=graph_construction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate BAP

log_file="graphgen151024.txt"

command="python dataprep2_graph_construction.py \
                --data_dir inference_test \
                --replace False \
                --masternode True \
                --protein_embeddings ankh_base esm2_t6 \
                --ligand_embeddings ChemBERTa_77M"
                
$command > $log_file 2>&1

# --protein_embeddings ankh_large ankh_base esm2_t6 esm2_t33 \