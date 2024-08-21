import h5py
import json
import numpy as np
from scipy.sparse import csr_matrix

# Initialize a log file that documents the removals
log_filename = "remove_redundancy.log"

# Import a list of the complex names from json file
with open('PDBbind_complexes.json', 'r') as f:
    complexes = json.load(f)

# Import the PDBbind data dictionary from json file
with open('PDBbind_data_dict.json', 'r') as f:
    pdbbind_data = json.load(f)

# Import a list of test complexes from json file
with open('PDBbind_data_splits/PDBbind_c0_data_split.json', 'r') as f:
    data_splits = json.load(f)
    casf2016 = data_splits['casf2016']
    casf2013 = data_splits['casf2013']
    test_dataset = casf2016 + casf2013
train_or_test = np.array([True if complex in test_dataset else False for complex in complexes])

# Vector showing which data point is in refined set with zeros and ones
refined = np.array([1 if "refined" in pdbbind_data[complex]['dataset'] else 0 for complex in complexes])

# Vector storing the resolutions of the complexes
resolutions = np.array([float(pdbbind_data[complex]['resolution']) if pdbbind_data[complex]['resolution'] != 'NMR' else 3.0 for complex in complexes])

# Pairwise label difference matrix
labels = np.array([pdbbind_data[complex]['log_kd_ki'] for complex in complexes])
pairwise_label_diff = labels[:, np.newaxis] - labels[np.newaxis, :]

# Import similarity matrix from hdf5 file
with h5py.File('pairwise_similarity_matrix.hdf5', 'r') as f:
    dset = f['distances']
    similarity_matrix = np.array(dset)

# Set all rows corresponding to test complexes to zero using train_or_test vector
similarity_matrix[train_or_test, :, :] = 0

# Set all matrix entries on the diagonal to zero
np.fill_diagonal(similarity_matrix[:,:,2], 0)


# Step 1: Apply the thresholds to create an adjacency matrix
mask1 = similarity_matrix[:,:,2] > 0.8 # Pairs with TM-score > 0.8
mask2 = similarity_matrix[:,:,0] + (1 - similarity_matrix[:,:,3]) > 0.8 # Pairs with S = tanimoto+(1-RMSE) > 0.8
mask3 = pairwise_label_diff < 1.0 # Pairs with label difference < 1.0
adjacency_matrix = mask1 & mask2 & mask3
adjacency_matrix = adjacency_matrix.astype(int)


# Step 2: Initialize the column sums
column_sums = np.sum(adjacency_matrix, axis=0)

# List to keep track of which data points remain
remaining_indices = np.arange(adjacency_matrix.shape[0])

while True:
    print()
    print("New removal iteration")
    print(f"Non-zero entries in adjacency_matrix: {np.count_nonzero(adjacency_matrix)}")

    # Step 3: Find the maximum sum and identify all indices with this sum
    max_sum_value = np.max(column_sums)
    print(f"Maximal sum: {max_sum_value}")
    if max_sum_value == 0: break # Break if all sums are zero (no more redundancies)
    
    # Print some information about the complexes with the maximum sum
    colsums = column_sums.tolist()
    info = [(sum, complexes[i], pdbbind_data[complexes[i]]['dataset'], resolutions[i])
            for i, sum in enumerate(colsums) if sum == max_sum_value]
    print(f"Info about the complexes with maximum similarities:\n{info}")

    # Step 4: Remove the data point with the maximum sum, preferring general complexes with higher resolution values
    max_indices = np.where(column_sums == max_sum_value)[0] # Select the columns with the maximum sum
    print(f"Maximal sum indices: {max_indices}")
    refined_or_general = refined[max_indices]
    print(f"Refined or general : {refined_or_general}")
    general = max_indices[refined_or_general == 0]
    if len(general) > 0: 
        print(f"General complexes found: {general}")
        max_indices = general # If there are some general complexes, select them
    max_sum_index = max_indices[np.argmax(resolutions[max_indices])] # Select the one with the highest resolution value
    print(f"Maximal sum index: {max_sum_index}, complex: {complexes[max_sum_index]}")
    

    # Step 6: Update the column sums incrementally
    similarities = adjacency_matrix[max_sum_index, :]
    print(similarities.tolist())
    print(f"Column sums before removal: {set(column_sums.tolist())}")
    column_sums -= similarities
    column_sums[max_sum_index] = 0  # Ensure the removed index is not considered again
    print(f"Column sums after removal: {set(column_sums.tolist())}")
    
    # Step 6: Zero out the row and column for the removed data point
    adjacency_matrix[max_sum_index, :] = 0
    adjacency_matrix[:, max_sum_index] = 0
    
    # # Record the removal of this data point
    membership = "refined" if refined[max_sum_index] == 1 else "general"
    similar_complexes_idx = np.where(similarities == 1)[0]
    print(similar_complexes_idx)
    similar_complexes_names = [complexes[idx] for idx in similar_complexes_idx]
    similar_complexes_resolutions = [resolutions[idx] for idx in similar_complexes_idx]
    similar_complexes_labels = [labels[idx] for idx in similar_complexes_idx]

    # similar_complexes = "\n---".join([f"{complex} (pK={label:.2f}, res={resolution:.2f})" for complex, label, resolution in zip(similar_complexes_names, similar_complexes_labels, similar_complexes_resolutions)])

    log_string = (
        f'Complex {complexes[max_sum_index]} ({membership}, pK={labels[max_sum_index]:.2f}, res={resolutions[max_sum_index]:.2f}) removed - '
        f'due to {len(similar_complexes_names)} similarities to \n---{similar_complexes_names}\n'
        )
    print(log_string)
    
    # with open(log_filename, 'a') as log_file: 
    #     log_file.write(log_string)

    remaining_indices = np.delete(remaining_indices, np.where(remaining_indices == max_sum_index)[0])

print(len(remaining_indices))