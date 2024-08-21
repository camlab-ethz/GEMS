import h5py
import numpy as np
import os
import json

# Get the test and training set complexes from the raw data folders
casf2016 = [folder for folder in os.listdir('PDBbind0/raw_data/CASF-2016/coreset')
            if len(folder) == 4 and folder[0].isdigit()]
casf2013 = [folder for folder in os.listdir('PDBbind0/raw_data/CASF-2013/coreset') 
            if len(folder) == 4 and folder[0].isdigit()]
test_dataset = casf2013 + casf2016

general = [folder for folder in os.listdir('PDBbind0/raw_data/v2020_general') if len(folder) == 4 and folder[0].isdigit()]
refined = [folder for folder in os.listdir('PDBbind0/raw_data/v2020_refined') if len(folder) == 4 and folder[0].isdigit()]

# Initialize the split of the filtered datasets c4
split_dict = {'casf2016': casf2016, 'casf2013': casf2013}

# Get the dict with the complexes' affinities from the json file
with open('PDBbind_data_dict.json', 'r') as f:
    affinity_data = json.load(f)

distance_matrix = 'pairwise_similarity_casf.hdf5'

# -------------------------------------------------------------------------------
# REMOVE THE TRAIN-TEST OVERLAP
# For each test complex, find highly similar training complexes and remove them
# -------------------------------------------------------------------------------

# Keep a log file with the reasons for the training set removals
log_file = open('dataset_filtering/C6/training_set_filtering.log', 'w')

# Get the indexes and names from json file
with open('pairwise_similarity_complexes.json', 'r') as f:
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
    in_casf2016 = test_complex in casf2016
    in_casf2013 = test_complex in casf2013

    membership = ""
    if in_casf2013: membership = membership + 'casf2013 '
    if in_casf2016: membership = membership + 'casf2016'

    # Load the pairwise similarity data
    with h5py.File(distance_matrix, 'r') as f:
        metrics = f['distances'][test_idx, :, :]

    test_complex_affinity = affinity_data[test_complex]['log_kd_ki']
    print(f"Processing {test_complex} with pK {test_complex_affinity} belonging to {membership}")
    log_file.write(f"---Processing {test_complex} with pK {test_complex_affinity}---\n")

    # Find the complexes that fulfill the following conditions:
    # - are in the training dataset 
    # - have TM-score higher than 0.8 to the test complex
    # - Tanimoto similarity and ligand positioning RMSD compared to the test complex fulfill the following formula:
    #   S = Tanimoto + (1 - RMSD) > 0.8
    
    # Extract the indexes of the complexes that satisfy the conditions, if there are any, else continue
    # mask = (train_or_test == 1) & (metrics[:, 0] > 0.4) & (metrics[:, 2] > 0.8) & (metrics[:, 3] < 0.5)

    mask1 = (metrics[:, 0] == 1.0)
    mask2 = (train_or_test == 1) & (metrics[:, 2] > 0.8) & (metrics[:, 0] + (1 - metrics[:, 3]) > 0.8)
    # Combine mask1 and mask2 with OR to get final mask
    mask = mask1 | mask2
    


    # Get the indexes and names of the complexes that satisfy the conditions
    similar_complexes_idx = mask.nonzero()[0]
    similar_complexes_names = [complexes[idx] for idx in similar_complexes_idx]

    # Check if the structurally similar training complexes have also similar affinity values
    similarities = []
    for complex, idx in zip(similar_complexes_names, similar_complexes_idx):
        
        # Check the difference between the affinity_values
        # If the absolute difference is lower than 1, remove the training complex
        dpK = np.abs(affinity_data[complex]['log_kd_ki'] - test_complex_affinity)
        if dpK < 1:

            similarities.append(complex)
            
            if complex in training_set_filtered:
                training_set_filtered.remove(complex)

                log_string = (
                    f'Complex {complex} removed - '
                    f'Tanimoto {metrics[idx, 0]:.2f} '
                    f'Seq_ID {metrics[idx, 1]:.2f} '
                    f'TMscore {metrics[idx, 2]:.2f} '
                    f'Mean dev {metrics[idx, 3]:.2f} '
                    f'S = {metrics[idx, 0] + (1 - metrics[idx, 3]):.2f} '
                    f'dpK {dpK:.2f} '
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

    # train_test_similarities[test_complex] = similarities
    # train_test_similarities_n[test_complex] = len(similarities)

print(len(training_set_filtered), len(casf2013_filtered), len(casf2016_filtered))


split_dict['train'] = training_set_filtered
split_dict['casf2016_c6'] = casf2016_filtered
split_dict['casf2013_c6'] = casf2013_filtered

with open('dataset_filtering/C6/PDBbind_c6_data_split.json', 'w', encoding='utf-8') as json_file:
    json.dump(split_dict, json_file, ensure_ascii=False, indent=4)

with open('dataset_filtering/C6/train_test_similarities_casf2016.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_test_sims_casf2016, json_file, ensure_ascii=False, indent=4)

with open('dataset_filtering/C6/train_test_similarities_casf2016_n.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_test_sims_casf2016_n, json_file, ensure_ascii=False, indent=4)

with open('dataset_filtering/C6/train_test_similarities_casf2013.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_test_sims_casf2013, json_file, ensure_ascii=False, indent=4)

with open('dataset_filtering/C6/train_test_similarities_casf2013_n.json', 'w', encoding='utf-8') as json_file:
    json.dump(train_test_sims_casf2013_n, json_file, ensure_ascii=False, indent=4)