import h5py
import numpy as np
import os
import json
import torch
import matplotlib.pyplot as plt


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, output, targets):
        return torch.sqrt(self.mse(output, targets))
criterion = RMSELoss()

def plot_predictions(y_true, y_pred, title, label):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, c='blue', label=label)
    axislim = 16
    plt.plot([0, axislim], [0, axislim], color='red', linestyle='--')
    plt.xlabel('True pK Values', fontsize=12)
    plt.ylabel('Predicted pK Values', fontsize=12)
    plt.ylim(0, axislim)
    plt.xlim(0, axislim)
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


# Add an argparse for remove_data_leakage and top_n
import argparse
parser = argparse.ArgumentParser(description="Compute and store pairwise metrics for 3D complexes.")
parser.add_argument('--data_split', required=True, type=str, help='Path to the data split dictionary to use for data leakage test')
parser.add_argument('--test_dataset', required=True, type=str, help='Name of the test dataset to use for data leakage test [casf2013, casf2016]')
parser.add_argument('--top_n', type=int, default=5, help='Number of top similar complexes to consider')
args = parser.parse_args()

top_n = args.top_n
data_split = args.data_split
test_dataset = args.test_dataset


# Import list of complexes from json file
with open('../PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_complexes.json', 'r') as f:
    complexes = json.load(f)

# Import affinity dict and get true affinity for each complex
with open('../PDBbind_data/PDBbind_data_dict.json', 'r') as f:
    affinity_data = json.load(f)

# Define the path to the pairwise similarity matrices (PSM)
PSM_tanimoto_file = '../PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tanimoto.hdf5'
PSM_tm_scores_file = '../PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_tm_scores.hdf5'
PSM_rmsd_file = '../PDBbind_data/similarity/pairwise_similarity_matrix/pairwise_similarity_rmsd_ligand.hdf5'


# Loop over the test complexes and look for the most similar training complexes
# ---------------------------------------------------------------------------------
print(f"Computing predictions for {test_dataset} test set\n\n")

with open(data_split, 'r') as f:
    split = json.load(f)
    train_dataset = split['train']
    test_complexes = split[test_dataset]

train_or_not = np.array([1 if complex in train_dataset else 0 for complex in complexes])

true_labels = [affinity_data[complex]['log_kd_ki'] for complex in test_complexes]
predicted_labels = {}

for complex in test_complexes:
    print()
    print(f"Finding similar training complexes for {complex}")
    complex_idx = complexes.index(complex)


    # Load the pairwise similarity data
    with h5py.File(PSM_tanimoto_file, 'r') as f:
        similarity_scores = f['similarities'][complex_idx, :]
    # Calculate similarity scores (Tanimoto)
    similarity_scores[complex_idx] = -np.inf # Set the metrics of the complex itself to small number
    similarity_scores[train_or_not == 0] = -np.inf # Set the metrics of all complexes not in the training dataset to small number


    sorted_indices = np.argsort(similarity_scores)
    sorted_indices = list(reversed(sorted_indices))
    print(f"Most similar indeces: {sorted_indices[0:5]}")
    print(f"Similarity scores: {similarity_scores[sorted_indices[0:5]]}")

    # Get the top n similar and average their labels
    top_indices = sorted_indices[:top_n]
    print(f"Most similar indeces: {top_indices}")
    names = [complexes[idx] for idx in top_indices]
    print(names)
    print(f"Tanimoto Scores: {[similarity_scores[idx] for idx in top_indices]}")    
    affinities = np.array([affinity_data[complex]['log_kd_ki'] for complex in names])
    print(f"Affinities: {affinities}")
    weights = similarity_scores[top_indices]
    print(f"Weights: {weights}")
    weighted_average = np.average(affinities, weights=weights)
    predicted_labels[complex] = weighted_average.item()


# Export the predictions to a json file
with open(f'{test_dataset}_predictions_top{top_n}_lig.json', 'w', encoding='utf-8') as json_file:
    json.dump(predicted_labels, json_file, ensure_ascii=False, indent=4)

# Compute the evaluation metrics
predicted_labels = np.array([predicted_labels[complex] for complex in test_complexes])
corr_matrix = np.corrcoef(true_labels, predicted_labels)
r = corr_matrix[0, 1]
rmse = criterion(torch.tensor(predicted_labels), torch.tensor(true_labels))

split_dict = os.path.basename(data_split).split('.')[0]
plot_predictions(true_labels, predicted_labels, f"{test_dataset} predictions \n{split_dict}\nWeighted average of labels of top {top_n} similar ligands\nR = {r:.3f}, RMSE = {rmse:.3f}", f"{test_dataset} Predictions")
plt.savefig(f'CASF2016_predictions_top{top_n}_lig', dpi=300)
