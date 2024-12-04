#!/bin/bash
#SBATCH --job-name=graph_construction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate BAP

log_file="graphgen_fill_up.txt"

command="python -m dataprep.graph_construction \
                --data_dir PDBbind_preprocessed \
                --replace False \
                --protein_embeddings ankh_base ankh_large esm2_t6 esm2_t33 \
                --ligand_embeddings ChemBERTa_77M"

# command="python -m dataprep.graph_construction \
#                 --data_dir inference_test \
#                 --replace False \
#                 --protein_embeddings ankh_base esm2_t6 \
#                 --ligand_embeddings ChemBERTa_77M"
                
$command > $log_file 2>&1