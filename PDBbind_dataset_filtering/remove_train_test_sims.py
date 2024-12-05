import h5py
import numpy as np
import os
import json

"""
This script removes data leakage by filtering out training complexes that are highly similar to test complexes 
based on pairwise similarity metrics. The filtered datasets are saved in a new split dictionary, including a filtered training dataset
and the filtered CASF2013 and CASF2016 test datasets (containing only the complexes with no similar complexes in the training set).

The script performs the following steps:
1. Load the names of all complexes in the training dataset and the test dataset (original split).
2. Initialize the split of the filtered dataset.
3. Load the precomputed pairwise similarity matrices (PSM).
4. Load the affinities of all complexes from a provided JSON file.
5. Iterate over the test complexes and find similar training complexes based on TM-score, Tanimoto similarity, 
    ligand positioning RMSD and affinity difference.
7. Remove training complexes that are highly similar to test complexes and log the reasons for removal.
8. Update the filtered datasets and save them in a new split dictionary.
9. Save the similarities found between test and training complexes in JSON files.

"""


# Define the test dataset complexes
with open('PDBbind_data/data_splits/PDBbind_data_split.json') as f:
    original_split_dict = json.load(f)
    casf2013 = original_split_dict['casf2013']
    casf2016 = original_split_dict['casf2016']
test_dataset = casf2013 + casf2016


# Initialize the split of the filtered dataset
split_dict = {'casf2016': casf2016, 'casf2013': casf2013}

# Get the dict with the complexes' affinities from the json file
with open('PDBbind_data/PDBbind_data_dict.json', 'r') as f:
    affinity_data = json.load(f)

# Define the path to the pairwise similarity matrices (PSM)
PSM_tanimoto_file = 'PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tanimoto.hdf5'
PSM_tm_scores_file = 'PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tm_scores.hdf5'
PSM_rmsd_file = 'PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_rmsd_ligand.hdf5'



# -------------------------------------------------------------------------------
# REMOVE THE TRAIN-TEST OVERLAP
# For each test complex, find highly similar training complexes and remove them
# Save the filtered datasets in a new split dict (json file)
# -------------------------------------------------------------------------------

# Keep a log file with the reasons for the training set removals
log_file = open('remove_train_test_sims.log', 'w')

# Get the indexes and names from json file
with open('PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_complexes.json', 'r') as f:
    complexes = json.load(f)


# Create list of test complexes and a list of training complexes
train_or_test = np.array([0 if complex in test_dataset else 1 for complex in complexes])
test_set = [(idx, complex) for idx, complex in enumerate(complexes) if train_or_test[idx] == 0] 
train_set = [(idx, complex) for idx, complex in enumerate(complexes) if train_or_test[idx] == 1]


# Keep track of the similarities found
train_test_similarities = {}
train_test_similarities_n = {}

train_test_sims_casf2016 = {}
train_test_sims_casf2016_n = {}

train_test_sims_casf2013 = {}
train_test_sims_casf2013_n = {}


# Initialize the filtered datasets
training_set_filtered = [complex for _, complex in train_set]
casf2013_filtered = [complex for _, complex in test_set if complex in casf2013]
casf2016_filtered = [complex for _, complex in test_set if complex in casf2016]

print(len(training_set_filtered), len(casf2013_filtered), len(casf2016_filtered))


# Iterate over the test complexes and look for similar training complexes
for test_idx, test_complex in test_set:

    # Check if the test complex is in casf2013 or casf2016 or both
    membership = ""
    if in_casf2013 := test_complex in casf2013:
        membership = membership + 'casf2013 '
    if in_casf2016 := test_complex in casf2016:
        membership = membership + 'casf2016'

    test_complex_affinity = affinity_data[test_complex]['log_kd_ki']
    
    print(f"Processing {test_complex} with pK {test_complex_affinity} belonging to {membership}")
    log_file.write(f"---Processing {test_complex} with pK {test_complex_affinity}---\n")

    # Find training complexes that fulfill the following conditions:
    # - have TM-score higher than 0.8 to the test complex
    # - Tanimoto similarity and ligand positioning RMSD compared to the test complex fulfill:
    #   Tanimoto + (1 - RMSD) > 0.8

    # Load the pairwise similarity data
    with h5py.File(PSM_tanimoto_file, 'r') as f:
        tanimoto = f['similarities'][test_idx, :]

    with h5py.File(PSM_tm_scores_file, 'r') as f:
        tm_scores = f['similarities'][test_idx, :]

    with h5py.File(PSM_rmsd_file, 'r') as f:
        rmsds = f['similarities'][test_idx, :]


    # Filter the training complexes that fulfill the conditions
    mask1 = (tanimoto > 0.9) | ((tm_scores > 0.8) & (tanimoto + (1 - rmsds) > 0.8))
    mask2 = (train_or_test == 1)
    mask = mask1 & mask2
    
    # Get the indexes and names of the complexes that satisfy the conditions
    similar_complexes_idx = mask.nonzero()[0]
    similar_complexes_names = [complexes[idx] for idx in similar_complexes_idx]


    similarities = []
    for complex, idx in zip(similar_complexes_names, similar_complexes_idx):
        
        # Check the difference between the affinity_values
        # If the absolute difference is lower than 1, remove the training complex
        dpK = np.abs(affinity_data[complex]['log_kd_ki'] - test_complex_affinity)
        if dpK < 1:

            if (tm_scores[idx] > 0.8) & (tanimoto[idx] + (1 - rmsds[idx]) > 0.8):
                reason = "COMPL SIMILARITY"
                similarities.append(complex)
            else:
                reason = "LIGND SIMILARITY"

            if complex in training_set_filtered:
                training_set_filtered.remove(complex)

                log_string = (
                    f'Complex {complex} removed due to {reason}- '
                    f'Tanimoto {tanimoto[idx]:.2f} '
                    f'TMscore {tm_scores[idx]:.2f} '
                    f'Ligand RMSD {rmsds[idx]:.2f} '
                    f'dpK {dpK:.2f} '
                    f'S = {tanimoto[idx] + (1 - rmsds[idx]) + tm_scores[idx] - dpK:.2f} '
                    f'to complex {test_complex} ({membership})\n'
                )
                log_file.write(log_string)


    # If similarites were found, remove the test complex from casf_filtered
    if len(similarities) > 0: 
        if test_complex in casf2013_filtered:
            casf2013_filtered.remove(test_complex)
        if test_complex in casf2016_filtered:
            casf2016_filtered.remove(test_complex)

    if in_casf2013:
        train_test_sims_casf2013[test_complex] = similarities
        train_test_sims_casf2013_n[test_complex] = len(similarities)
    if in_casf2016:
        train_test_sims_casf2016[test_complex] = similarities
        train_test_sims_casf2016_n[test_complex] = len(similarities)


print(len(training_set_filtered), len(casf2013_filtered), len(casf2016_filtered))


split_dict['train'] = training_set_filtered
split_dict['casf2016_c7'] = casf2016_filtered
split_dict['casf2013_c7'] = casf2013_filtered

with open('PDBbind_split_leakage_removed.json', 'w', encoding='utf-8') as json_file:
    json.dump(split_dict, json_file, ensure_ascii=False, indent=4)

with open('train_test_similarities_casf2016.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_test_sims_casf2016, json_file, ensure_ascii=False, indent=4)

with open('train_test_similarities_casf2016_n.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_test_sims_casf2016_n, json_file, ensure_ascii=False, indent=4)

with open('train_test_similarities_casf2013.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_test_sims_casf2013, json_file, ensure_ascii=False, indent=4)

with open('train_test_similarities_casf2013_n.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_test_sims_casf2013_n, json_file, ensure_ascii=False, indent=4)
