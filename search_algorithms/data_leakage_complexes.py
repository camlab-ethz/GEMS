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
parser.add_argument('--data_split', type=str, default='c0', help='Data split to use for data leakage test')
parser.add_argument('--top_n', type=int, default=5, help='Number of top similar complexes to consider')
args = parser.parse_args()

top_n = args.top_n
data_split = args.data_split
if data_split == 'c0': title = '(With Data Leakage)'
else: title = '(Data Leakage Removed)'


# Import list of complexes from json file
with open('PDBbind_complexes.json', 'r') as f:
    complexes = json.load(f)

with open(f'PDBbind_data_splits/PDBbind_{data_split}_data_split.json', 'r') as f:
    data_splits = json.load(f)
    train_dataset = data_splits['train']
    casf2016 = data_splits['casf2016']
    casf2013 = data_splits['casf2013']
    test_dataset = casf2016 + casf2013

train_or_not = np.array([1 if complex in train_dataset else 0 for complex in complexes])

# Import affinity dict and get true affinity for each complex
with open('PDBbind_data_dict.json', 'r') as f:
    affinity_data = json.load(f)

distance_matrix = 'pairwise_similarity_matrix/pairwise_similarity_matrix_optimal.hdf5'

true_labels_casf2016 = [affinity_data[complex]['log_kd_ki'] for complex in casf2016]
true_labels_casf2013 = [affinity_data[complex]['log_kd_ki'] for complex in casf2013]


# Loop over the test complexes and look for the most similar training complexes
# ---------------------------------------------------------------------------------


print(f"Computing predictions for CASF2016 test set\n\n")

predicted_labels_casf2016 = []
for complex in casf2016:
    print(f"Finding similar training complexes for {complex}")
    complex_idx = complexes.index(complex)

    # Get the similarity data to all training complexes
    with h5py.File(distance_matrix, 'r') as f:
        metrics = f['distances'][complex_idx, :, :]
        metrics[complex_idx, :] = 0 # Set the metrics of the complex itself to zero
        metrics[train_or_not == 0, :] = 0 # Set the metrics of all complexes not in the training dataset to zero using the train_or_test mask

    # Calculate similarity scores
    similarity_scores = np.sum(metrics[:, :3], axis=1) - metrics[:, 3]
    sorted_indices = np.argsort(similarity_scores)
    sorted_indices = list(reversed(sorted_indices))

    # Get the top n similar complexes and average their labels
    top_indices = sorted_indices[:top_n]
    names = [complexes[idx] for idx in top_indices]
    affinities = np.array([affinity_data[complex]['log_kd_ki'] for complex in names])
    weights = similarity_scores[top_indices]
    weighted_average = np.average(affinities, weights=weights)
    predicted_labels_casf2016.append(weighted_average.item())
    print(weights)

# Compute the evaluation metrics
predicted_labels_casf2016 = np.array(predicted_labels_casf2016)
corr_matrix = np.corrcoef(true_labels_casf2016, predicted_labels_casf2016)
r = corr_matrix[0, 1]
rmse = criterion(torch.tensor(predicted_labels_casf2016), torch.tensor(true_labels_casf2016))

plot_predictions(true_labels_casf2016, predicted_labels_casf2016, f"CASF2016 Predictions {title}\nWeighted average of labels of top {top_n} similar complexes\nR = {r:.3f}, RMSE = {rmse:.3f}", "CASF-2016 Predictions")
plt.savefig(f'data_leakage_test/CASF2016_complexes_{data_split}_top{top_n}', dpi=300)




print(f"\n\nComputing predictions for CASF2013 test set\n\n")

predicted_labels_casf2013 = []
for complex in casf2013:
    print(f"Finding similar training complexes for {complex}")
    
    complex_idx = complexes.index(complex)

    # Get the similarity data to all training complexes
    with h5py.File(distance_matrix, 'r') as f:
        metrics = f['distances'][complex_idx, :, :]
        metrics[complex_idx, :] = 0 # Set the metrics of the complex itself to zero
        metrics[train_or_not == 0, :] = 0 # Set the metrics of all complexes not in the training dataset to zero using the train_or_test mask

    # Calculate similarity scores
    similarity_scores = np.sum(metrics[:, :3], axis=1) - metrics[:, 3]
    sorted_indices = np.argsort(similarity_scores)
    sorted_indices = list(reversed(sorted_indices))

    # Get the top n similar complexes and average their labels
    top_indices = sorted_indices[:top_n]
    names = [complexes[idx] for idx in top_indices]
    affinities = np.array([affinity_data[complex]['log_kd_ki'] for complex in names])
    weights = similarity_scores[top_indices] + 0.01
    print(weights)
    weighted_average = np.average(affinities, weights=weights)
    predicted_labels_casf2013.append(weighted_average.item())

# Compute the evaluation metrics
predicted_labels_casf2013 = np.array(predicted_labels_casf2013)
corr_matrix = np.corrcoef(true_labels_casf2013, predicted_labels_casf2013)
r = corr_matrix[0, 1]
rmse = criterion(torch.tensor(predicted_labels_casf2013), torch.tensor(true_labels_casf2013))

plot_predictions(true_labels_casf2013, predicted_labels_casf2013, f"CASF2013 Predictions {title}\nWeighted average of labels of top {top_n} similar complexes\nR = {r:.3f}, RMSE = {rmse:.3f}", "CASF-2013 Predictions")
plt.savefig(f'data_leakage_test/CASF2013_complexes_{data_split}_top{top_n}', dpi=300)