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
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, c='blue', label=label)
    axislim = 16
    plt.plot([0, axislim], [0, axislim], color='red', linestyle='--')
    plt.xlabel('True pK Values')
    plt.ylabel('Predicted pK Values')
    plt.ylim(0, axislim)
    plt.xlim(0, axislim)
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.show()


# Add an argparse for remove_data_leakage and top_n
import argparse
parser = argparse.ArgumentParser(description="Compute and store pairwise metrics for 3D complexes.")
parser.add_argument('--remove_data_leakage', default=False, action='store_true', help='Remove data leakage from the training set')
parser.add_argument('--top_n', type=int, default=5, help='Number of top similar complexes to consider')
args = parser.parse_args()

remove_data_leakage = args.remove_data_leakage
top_n = args.top_n
print(f"Remove data leakage: {remove_data_leakage}", flush=True)


# Import list of complexes from json file
with open('PDBbind_complexes.json', 'r') as f:
    complexes = json.load(f)

# Import a list of test complexes from json file
with open('PDBbind_data_splits/PDBbind_c0_data_split.json', 'r') as f:
    data_splits = json.load(f)
    casf2016 = data_splits['casf2016']
    casf2013 = data_splits['casf2013']
    test_dataset = casf2016 + casf2013

train_or_test = np.array([0 if complex in test_dataset else 1 for complex in complexes])

# Import affinity dict and get true affinity for each complex
with open('PDBbind_data_dict.json', 'r') as f:
    affinity_data = json.load(f)

distance_matrix = 'pairwise_similarity_casf.hdf5'

true_labels_casf2016 = [affinity_data[complex]['log_kd_ki'] for complex in casf2016]
true_labels_casf2013 = [affinity_data[complex]['log_kd_ki'] for complex in casf2013]


### Loop over the test complexes and look for the most similar training complexes
# ---------------------------------------------------------------------------------

print(f"Computing predictions for CASF2016 test set\n\n")

# Predict a mean label for each test complex with a tiny bit of noise
predicted_labels_casf2016 = [np.mean(true_labels_casf2016) + np.random.normal(0, 0.01) for _ in range(len(true_labels_casf2016))]
#predicted_labels_casf2016 = [np.mean(true_labels_casf2016)] * len(true_labels_casf2016)


# Compute the evaluation metrics
predicted_labels_casf2016 = np.array(predicted_labels_casf2016)
corr_matrix = np.corrcoef(true_labels_casf2016, predicted_labels_casf2016)
r = corr_matrix[0, 1]
rmse = criterion(torch.tensor(predicted_labels_casf2016), torch.tensor(true_labels_casf2016))

# Plot the predictions for test data
if remove_data_leakage:
    plot_predictions(true_labels_casf2016, predicted_labels_casf2016, f"CASF2016 Predictions (data leakage REMOVED c6)\nWeighted average of labels of top {top_n} similar complexes\nR = {r:.3f}, RMSE = {rmse:.3f}", "Test Predictions")
    plt.savefig(f'data_leakage_test/CASF2016_complexes_removed_c7_top{top_n}', dpi=300)
else:
    plot_predictions(true_labels_casf2016, predicted_labels_casf2016, f"CASF2016 Predictions (with data leakage)\nWeighted average of labels of top {top_n} similar complexes\nR = {r:.3f}, RMSE = {rmse:.3f}", "Test Predictions")
    plt.savefig(f'data_leakage_test/CASF2016_mean_noisy', dpi=300)