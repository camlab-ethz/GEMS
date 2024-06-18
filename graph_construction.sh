#!/bin/bash
#SBATCH --job-name=graph_construction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate graphgen

log_file="graphgen180624.txt"

command="python graph_construction.py \
                --data_dir PDBbind_test \
                --replace True \
                --masternode True \
                --protein_embeddings ankh_large ankh_base esm2_t6 esm2_t33 \
                --ligand_embeddings ChemBERTa_10M"
                
$command > $log_file 2>&1