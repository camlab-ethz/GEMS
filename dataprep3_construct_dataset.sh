#!/bin/bash
#SBATCH --job-name=dataset_construction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

module load eth_proxy

source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate BAP

# command="python dataprep3_construct_dataset.py \
#                 --data_dir inference_test \
#                 --protein_embeddings ankh_base esm2_t6 \
#                 --ligand_embeddings ChemBERTa_77M \
#                 --data_dict PDBbind_data/PDBbind_data_dict.json
#                 --save_path inference_test/dataset_inference_151024.pt"

# command="python dataprep3_construct_dataset.py \
#                 --data_dir inference_test \
#                 --protein_embeddings ankh_base esm2_t6 \
#                 --ligand_embeddings ChemBERTa_77M \
#                 --save_path dataset_inference_051124.pt"

# command="python -m dataprep.construct_dataset \
#                 --data_dir inference_test \
#                 --protein_embeddings ankh_base esm2_t6 \
#                 --ligand_embeddings ChemBERTa_77M \
#                 --save_path dataset_inference_all.pt"

# command="python -m dataprep.construct_dataset \
#                 --data_dir inference_test \
#                 --save_path dataset_inference_none.pt"

command="python -m dataprep.construct_dataset \
                --data_dir inference_test \
                --ligand_embeddings ChemBERTa_77M \
                --save_path dataset_inference_lig.pt"

$command