import h5py
import json
import numpy as np

input_data_split = 'PDBbind_data_splits/PDBbind_c0_data_split.json'
output_data_split = 'PDBbind_data_splits/PDBbind_c8_data_split.json'

# Thresholds for filtering based on TM-scores, S = tanimoto+(1-RMSE) and label differences
TM_threshold = 0.8
S_threshold = 0.8
label_threshold = 1.0

# Initialize log file
log_filename = 'remove_redundancy.log'

# Import a list of the complex names in the database
with open('PDBbind_complexes.json', 'r') as f:
    complexes = json.load(f)

# INPUT TRAINING DATASET from json file
with open(input_data_split, 'r') as f:
    data_splits = json.load(f)
    casf2016 = data_splits['casf2016']
    casf2013 = data_splits['casf2013']
    test_dataset = casf2016 + casf2013
    train_dataset = data_splits['train']
train_or_test = np.array([True if complex in test_dataset else False for complex in complexes])

# OUTPUT TRAINING DATASET
train_dataset_filtered = train_dataset.copy()

# Write the initial information to the log file

print("Removing redundancy from the training dataset\n")
print(f"Input data split: {input_data_split}\n")
print(f"Initial number of training complexes: {len(train_dataset)}\n")
print(f"Thresholds: \nTM-score > {TM_threshold}, \nS = tanimoto+(1-RMSE) > {S_threshold}, \nlabel difference < {label_threshold}\n")
print("\n")

# Import the PDBbind data dictionary from json file
with open('PDBbind_data_dict.json', 'r') as f:
    pdbbind_data = json.load(f)
    
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

# Make the similarity matrix symmetric in case there are some numerical errors
for dim in range(similarity_matrix.shape[2]):
    similarity_matrix[:, :, dim] = np.maximum(similarity_matrix[:, :, dim], similarity_matrix[:, :, dim].T)

# Set all rows and columns corresponding to test complexes to zero using train_or_test vector
similarity_matrix[train_or_test, :, :] = 0
similarity_matrix[:, train_or_test, :] = 0

# Set all rows and columns corresponding to complexes NOT in the current training dataset to zero
train_dataset_indices = [complexes.index(complex) for complex in train_dataset if complex in complexes]
non_train_dataset_indices = [complexes.index(complex) for complex in complexes if complex not in train_dataset]
similarity_matrix[non_train_dataset_indices, :, :] = 0
similarity_matrix[:, non_train_dataset_indices, :] = 0

# Step 1: Apply the thresholds to create an adjacency matrix
mask1 = similarity_matrix[:,:,2] > TM_threshold # Pairs with TM-score > 0.8
mask2 = similarity_matrix[:,:,0] + (1 - similarity_matrix[:,:,3]) > S_threshold # Pairs with S = tanimoto+(1-RMSE) > 0.8
mask3 = pairwise_label_diff < label_threshold # Pairs with small label difference
adjacency_matrix = mask1 & mask2 & mask3
adjacency_matrix = adjacency_matrix.astype(int)
adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T) # Make the adjacency matrix symmetric

# Make sure all matrix entries on the diagonal are zero
np.fill_diagonal(adjacency_matrix, 0)

# Step 2: Initialize the column sums
column_sums = np.sum(adjacency_matrix, axis=0)

# Compute some metrics to describe the redundancy (cluster sizes, number of clusters, etc.)
cluster_sizes = []
print(f"Number of complexes not in any cluster: {len(train_dataset_indices) - np.count_nonzero(column_sums)}/{len(train_dataset_indices)}")
print(f"Number of complexes in clusters: {np.count_nonzero(column_sums)}/{len(train_dataset_indices)}")

# Average number of similarities per complex (excluding zeros)
print(f"Average number of similarities per complex: {np.mean(column_sums[column_sums > 0]):.2f}")

# Median number of similarities per complex (excluding zeros)
print(f"Median number of similarities per complex: {np.median(column_sums[column_sums > 0]):.2f}")

# Maximum number of similarities
print(f"Maximum number of similarities: {np.max(column_sums)}")



# # List to keep track of which data points remain
# remaining_indices = np.arange(adjacency_matrix.shape[0])

# while True:

#     # Step 3: Find the maximum sum and identify all indices with this sum
#     max_sum_value = np.max(column_sums)
#     if max_sum_value == 0: break # Break if all sums are zero (no more redundancies)
    
#     # Save some information about the complexes with the maximum sum
#     info = [(i, sum, complexes[i], labels[i], pdbbind_data[complexes[i]]['dataset'], resolutions[i])
#             for i, sum in enumerate(column_sums) if sum == max_sum_value]    

#     # Step 4: Remove the data point with the maximum sum, preferring general complexes with higher resolution values
#     max_indices = np.where(column_sums == max_sum_value)[0] # Select the columns with the maximum sum
#     refined_or_general = refined[max_indices]
#     general = max_indices[refined_or_general == 0]
#     if len(general) > 0: max_indices = general # If there are some general complexes, select them
#     to_remove_idx = max_indices[np.argmax(resolutions[max_indices])] # Select the one with the highest resolution value

#     # Step 5: Update the column sums incrementally
#     similarities = adjacency_matrix[to_remove_idx, :].copy()

#     # # Print some information about the similarities
#     # before = sorted([(column_sums[i], similarities[i]) for i in range(len(column_sums)) if similarities[i] > 0 or column_sums[i] == max_sum_value], reverse=True)
#     # print(f"Column sums before removal: {[tup[0] for tup in before]}")
#     # print(f"Similarities vector:        {[tup[1] for tup in before]}")
    
#     # Update the column sums
#     column_sums -= similarities
#     column_sums[to_remove_idx] = 0  # Ensure the removed index is not considered again

#     # # Print some information about the similarities after the removal
#     # after = sorted([(column_sums[i], similarities[i]) for i in range(len(column_sums)) if similarities[i] > 0], reverse=True)
#     # print(f"Column sums after removal:  {[tup[0] for tup in after]}")

#     # Step 6: Zero out the row and column for the removed data point
#     adjacency_matrix[to_remove_idx, :] = 0
#     adjacency_matrix[:, to_remove_idx] = 0
    

#     # Record the removal of this data point
#     with open(log_filename, 'a') as log_file:
#         print("New removal iteration" + "\n")
#         print(f"Non-zero entries in adjacency_matrix: {np.count_nonzero(adjacency_matrix)}" + "\n")
#         print(f"Maximal number of similarities: {max_sum_value}" + "\n")
#         print(f"Info about the complexes with maximum similarities:" + "\n")
#         for inf in info: print(str(inf) + "\n")
#         print(f"To remove index: {to_remove_idx}, complex: {complexes[to_remove_idx]}" + "\n")


#     membership = "refined" if refined[to_remove_idx] == 1 else "general"
#     similar_complexes_idx = np.nonzero(similarities)[0]


#     # If the complex is still in the current training dataset, remove it
#     if complexes[to_remove_idx] in train_dataset_filtered:
#         train_dataset_filtered.remove(complexes[to_remove_idx])
        
#         log_string = (
#             f'Complex {complexes[to_remove_idx]} ({membership}, pK={labels[to_remove_idx]:.2f}, res={resolutions[to_remove_idx]:.2f}) removed - '
#             f'due to {len(similar_complexes_idx)} similarities:'
#         )

#         print(log_string)

#         for ind in similar_complexes_idx:
#             log_string += f'\n---{complexes[ind]} (Label:{labels[ind]:.2f} +-{pairwise_label_diff[to_remove_idx, ind]:.2f} with Tanimoto:{similarity_matrix[to_remove_idx, ind, 0]:.2f}, TM-score:{similarity_matrix[to_remove_idx, ind, 2]:.2f}, RMSD:{similarity_matrix[to_remove_idx, ind, 3]:.2f})'
  
#         with open(log_filename, 'a') as log_file:
#             print(log_string+ "\n\n")

#     else:
#         with open(log_filename, 'a') as log_file:
#             print(f"Complex {complexes[to_remove_idx]} is not in the current training dataset\n")


# # Write results to log file
# with open(log_filename, 'a') as log_file:
#     print(f"\nFinal number of training complexes: {len(train_dataset_filtered)}\n")

# # Copy the input data split
# data_splits_filtered = data_splits.copy()
# data_splits_filtered['train'] = train_dataset_filtered
# with open(output_data_split, 'w') as f:
#     json.dump(data_splits_filtered, f, indent=4)