## Dataset Filtering Instructions

Our filtering algorithm was designed to remove train-test similarites from PDBbind by excluding training complexes that are very similar to the test complexes in the CASF datasets. 
This leaves the CASF test datasets unchanged, but turns them into more independent test datasets for models trained on the filtered PDBbind training dataset. 
In a second optional step, the algorithm removes training dataset redundancy based on predifined similarity thresholds. 
The scripts for removing train-test similarities and training redundancy can be found at `PDBbind_dataset_filtering/remove_train_test_sims.py`
and `PDBbind_dataset_filtering/remove_train_redundancy.py`

## Removing Train-Test Similarities

This script iterates over all test complexes and identifies highly similar training complexes based on the affinity difference and pairwise similarity matrices generated for Tanimoto similarity, TM-scores, and ligand RMSDs.
The most similar complexes are removed from the training dataset based on predefined thresholds. In order to make the ligands in the test dataset unique, the script removes all training complexes that have a nearly identical ligand (and similar affinity label) to a test complex.

### Inputs to the filtering algorithm `remove_train_test_sims.py`:
* **Input Split:** The original split of the dataset, a dictionary assigning all complexes to 'train', 'casf2016' and 'casf2013'.
    * see `PDBbind_data/PDBbind_data_split_pdbbind.json`
* **Pairwise Similarity Matrices:** containing all pairwise Tanimoto Similarites, TM-Scores and Ligand RMSDs for the whole PDBbind dataset (including CASF).
    * see `PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tanimoto.hdf5`
    * see `PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tm_scores.hdf5`
    * see `PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_rmsd_ligand.hdf5`
* **List of Indexes:** A list of datapoint IDs corresponding to the order of the complexes in the pairwise similarity matrices.
    * see `PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_complexes.json`
* **PDBbind Affinity Data:** A dictionary containing log_kd_ki values for all complexes in PDBbind.
    * see `PDBbind_data/PDBbind_data_dict.json`
 
#### Outputs of the filtering algorithm `remove_train_test_sims.py`:
* **Output Split:** The updated split of the dataset, with a potentially smaller 'train' dataset.
    * see `PDBbind_data/PDBbind_data_split_cleansplit.json`
* **Log Files:** A log file and some additional dictionaries containing information on the similarities to the training set for each test complex.

To run this code, change the paths in the "INPUTS" section of the code to your needs.



## Removing Training Dataset Redundancy

This script generates a symmetric adjacency matrix to represent pairwise similarites between the PDBbind training complexes based on multiple similarity thresholds (TM-score, Tanimoto similarity, RMSD, and label differences). The code iteratively removes complexes from the training dataset until all columns sums of the adjacency matrix are zero, removing all pairwise similarities based on the given thresholds. During this process, complexes belonging to the refined PDBbind dataset and complexes with high resolution are kept in the dataset when possible.

### Inputs to the filtering algorithm `remove_train_redundancy.py`:
* **Input Training Dataset:** The original training dataset, including a list of all complex IDs that are in the original training dataset (can be the output of the `remove_train_test_sims.py` script).
    * see `PDBbind_data/PDBbind_data_split_pdbbind.json`
* **Pairwise Similarity Matrices:** containing all pairwise Tanimoto Similarites, TM-Scores and Ligand RMSDs for the whole PDBbind dataset (including CASF).
    * see `PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tanimoto.hdf5`
    * see `PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tm_scores.hdf5`
    * see `PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_rmsd_ligand.hdf5`
* **List of Indexes:** A list of datapoint IDs corresponding to the order of the complexes in the pairwise similarity matrices.
    * see `PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_complexes.json`
* **PDBbind Affinity, Resolution and Dataset Membership Data:** A dictionary containing log_kd_ki values for all complexes in PDBbind, as well as their resolution and their membership in the refined dataset.
    * see `PDBbind_data/PDBbind_data_dict.json`
 
To run this code, change the paths in the "INPUTS" section of the code to your needs.

