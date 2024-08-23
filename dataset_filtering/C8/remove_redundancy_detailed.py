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

# # Pairwise label difference matrix
labels = np.array([pdbbind_data[complex]['log_kd_ki'] for complex in complexes])
pairwise_label_diff = labels[:, np.newaxis] - labels[np.newaxis, :]
pairwise_label_diff = np.abs(pairwise_label_diff)

# # Import similarity matrix from hdf5 file
with h5py.File('pairwise_similarity_matrix.hdf5', 'r') as f:
    dset = f['distances']
    similarity_matrix = np.array(dset)

# Make the similarity matrix symmetric
for dim in range(similarity_matrix.shape[2]):
    similarity_matrix[:, :, dim] = np.maximum(similarity_matrix[:, :, dim], similarity_matrix[:, :, dim].T)

# Set all rows corresponding to test complexes to zero using train_or_test vector
similarity_matrix[train_or_test, :, :] = 0
similarity_matrix[:, train_or_test, :] = 0


# Step 1: Apply the thresholds to create an adjacency matrix
mask1 = similarity_matrix[:,:,2] > 0.8 # Pairs with TM-score > 0.8
mask2 = similarity_matrix[:,:,0] + (1 - similarity_matrix[:,:,3]) > 1.0 # Pairs with S = tanimoto+(1-RMSE) > 0.8
mask3 = pairwise_label_diff < 0.5 # Pairs with small label difference
adjacency_matrix = mask1 & mask2 & mask3
adjacency_matrix = adjacency_matrix.astype(int)
print(f"Adjacency Matrix is symmetric: {np.array_equal(adjacency_matrix, adjacency_matrix.T)}")
adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T) # Make the adjacency matrix symmetric
print(f"Shape of adjacency matrix: {adjacency_matrix.shape}")
print(f"Adjacency Matrix is symmetric: {np.array_equal(adjacency_matrix, adjacency_matrix.T)}")

# # Define a 10x10 example adjacency matrix
# adjacency_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
#                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                              [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Set all matrix entries on the diagonal to zero
np.fill_diagonal(adjacency_matrix, 0)

# Step 2: Initialize the column sums
column_sums = np.sum(adjacency_matrix, axis=0)
print(f"Column sums: {column_sums}, {column_sums.dtype}")

# List to keep track of which data points remain
remaining_indices = np.arange(adjacency_matrix.shape[0])

while True:
    print()
    print("New removal iteration")
    print(f"Non-zero entries in adjacency_matrix: {np.count_nonzero(adjacency_matrix)}")

    # Step 3: Find the maximum sum and identify all indices with this sum
    max_sum_value = np.max(column_sums)
    print(f"Maximal number of similarities: {max_sum_value}")
    if max_sum_value == 0: break # Break if all sums are zero (no more redundancies)
    
    # Print some information about the complexes with the maximum sum
    info = [(i, sum, complexes[i], labels[i], pdbbind_data[complexes[i]]['dataset'], resolutions[i])
            for i, sum in enumerate(column_sums) if sum == max_sum_value]
    print(f"Info about the complexes with maximum similarities:")
    for inf in info: print(inf)

    # Step 4: Remove the data point with the maximum sum, preferring general complexes with higher resolution values
    max_indices = np.where(column_sums == max_sum_value)[0] # Select the columns with the maximum sum

    refined_or_general = refined[max_indices]
    general = max_indices[refined_or_general == 0]
    if len(general) > 0: max_indices = general # If there are some general complexes, select them
    to_remove_idx = max_indices[np.argmax(resolutions[max_indices])] # Select the one with the highest resolution value
    print(f"To remove index: {to_remove_idx}, complex: {complexes[to_remove_idx]}")

    # Step 6: Update the column sums incrementally
    similarities = adjacency_matrix[to_remove_idx, :].copy()

    # indexes of values in column_sums that are greater than 10
    before = sorted([(column_sums[i], similarities[i]) for i in range(len(column_sums)) if similarities[i] > 0 or column_sums[i] == max_sum_value], reverse=True)
    print(f"Column sums before removal: {[tup[0] for tup in before]}")
    print(f"Similarities vector:        {[tup[1] for tup in before]}")
    column_sums -= similarities
    column_sums[to_remove_idx] = 0  # Ensure the removed index is not considered again
    after = sorted([(column_sums[i], similarities[i]) for i in range(len(column_sums)) if similarities[i] > 0], reverse=True)
    print(f"Column sums after removal:  {[tup[0] for tup in after]}")

    # Step 6: Zero out the row and column for the removed data point
    adjacency_matrix[to_remove_idx, :] = 0
    adjacency_matrix[:, to_remove_idx] = 0
    
    # # Record the removal of this data point
    membership = "refined" if refined[to_remove_idx] == 1 else "general"

    similar_complexes_idx = np.nonzero(similarities)[0]
    similar_complexes_names = [complexes[idx] for idx in similar_complexes_idx]
    similar_complexes_resolutions = [resolutions[idx] for idx in similar_complexes_idx]
    similar_complexes_labels = [labels[idx] for idx in similar_complexes_idx]


    log_string = (
        f'Complex {complexes[to_remove_idx]} ({membership}, pK={labels[to_remove_idx]:.2f}, res={resolutions[to_remove_idx]:.2f}) removed - '
        f'due to {len(similar_complexes_names)} similarities to'
    )
    for ind in similar_complexes_idx:
        log_string += f'\n---{complexes[ind]} (Label:{labels[ind]:.2f} +-{pairwise_label_diff[to_remove_idx, ind]:.2f} with Tanimoto:{similarity_matrix[to_remove_idx, ind, 0]:.2f}, TM-score:{similarity_matrix[to_remove_idx, ind, 2]:.2f}, RMSD:{similarity_matrix[to_remove_idx, ind, 3]:.2f})'
    
    print(log_string)
    
#     # with open(log_filename, 'a') as log_file: 
#     #     log_file.write(log_string)

    remaining_indices = np.delete(remaining_indices, np.where(remaining_indices == to_remove_idx)[0])

print(len(remaining_indices))