import argparse
import sys
import csv
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from model.GATE18 import *


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, output, targets):
        return torch.sqrt(self.mse(output, targets))


def load_model_state(model, state_dict_path, device):
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(state_dict_path))
        model.eval()  # Set the model to evaluation mode
    return model



# Evaluation Function
#-------------------------------------------------------------------------------------------------------------------------------
def evaluate(models, loader, criterion, device, labels):
    
    # Initialize variables to accumulate the evaluation results
    total_loss = 0.0
    y_true = []
    y_pred = []
    id = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for graphbatch in loader:
            graphbatch.to(device)
            targets = graphbatch.y

            # Forward pass ENSEMBLE MODEL
            outputs = []
            for model in models: 
                outputs.append(model(graphbatch).view(-1))
            output = torch.mean(torch.stack(outputs), dim=0)    
            loss = criterion(output, targets)

            # Accumulate loss and collect the true and predicted values for later use
            total_loss += loss.item()
            y_true.extend(targets.tolist())
            y_pred.extend(output.tolist())
            id.extend(graphbatch.id)


    if labels:
        # Calculate evaluation metrics
        eval_loss = total_loss / len(loader)

        # Pearson Correlation Coefficient
        corr_matrix = np.corrcoef(y_true, y_pred)
        r = corr_matrix[0, 1]

        # R2 Score
        r2_score = 1 - np.sum((np.array(y_true) - np.array(y_pred)) ** 2) / np.sum((np.array(y_true) - np.mean(np.array(y_true))) ** 2)

        # RMSE in pK unit
        min=0
        max=16
        true_labels_unscaled = torch.tensor(y_true) * (max - min) + min
        predictions_unscaled = torch.tensor(y_pred) * (max - min) + min
        rmse = criterion(predictions_unscaled, true_labels_unscaled)
        return eval_loss, r, rmse, r2_score, true_labels_unscaled, predictions_unscaled, id
    
    else:
        min=0
        max=16
        true_labels_unscaled = torch.tensor(y_true)
        predictions_unscaled = torch.tensor(y_pred) * (max - min) + min
        return true_labels_unscaled, predictions_unscaled, id
#-------------------------------------------------------------------------------------------------------------------------------


# Plotting Functions
#-------------------------------------------------------------------------------------------------------------------------
def plot_error_histogram(ax, errors, title):
    n, bins, patches = ax.hist(errors, bins=50, color='blue', edgecolor='black')
    
    # Add text on top of each column
    for count, patch in zip(n, patches):
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f'{int(count)}', 
                ha='center', va='bottom')

    ax.set_title(title)
    ax.set_xlabel('Absolute Error (pK)')
    ax.set_ylabel('Frequency')


def plot_predictions(ax, y_true, y_pred, title, label):
    ax.scatter(y_true, y_pred, alpha=0.5, c='blue', label=label)
    axislim = 16
    ax.plot([0, axislim], [0, axislim], color='red', linestyle='--')
    ax.set_xlabel('True pK Values')
    ax.set_ylabel('Predicted pK Values')
    ax.set_ylim(0, axislim)
    ax.set_xlim(0, axislim)
    ax.axhline(0, color='grey', linestyle='--')
    ax.axvline(0, color='grey', linestyle='--')
    ax.set_title(title)
    ax.legend()
#-------------------------------------------------------------------------------------------------------------------------



def parse_args():
    parser = argparse.ArgumentParser(description="Testing Parameters and Input Dataset Control")

    # Model Parameters
    parser.add_argument("--dataset_path", required=True, help="The path to the test dataset pt file")
    return parser.parse_args()

args = parse_args()

dataset_path = args.dataset_path


# Load the dataset
print(f"Loading dataset from {dataset_path}")
dataset = torch.load(dataset_path)
node_feat_dim = dataset[0].x.shape[1]
edge_feat_dim = dataset[0].edge_attr.shape[1]
print(f"Dataset Loaded with {len(dataset)} samples")

# Check if the dataset has labels
labels = dataset[0].y > 0
print(f"Dataset has labels: {labels}")


# SELECT THE MODEL ARCHITECTURE AND STATE DICT PATHS
#------------------------------------------------------------------------------------------------------
# Check if the dataset has protein and ligand embeddings and choose the correct model state dict paths
protein_embeddings = dataset.protein_embeddings
ligand_embeddings = dataset.ligand_embeddings

# Currently providing stdicts for the following combinations of embeddings:
# 1. No embeddings (00AEPL with GATE18e)
# 2. ChemBERTa-77M only (00AEPL with GATE18d)
# 3. ChemBERTa-77M and ankh_base (B0AEPL with GATE18d)
# 4. ChemBERTa-77M and esm2_t6 (06AEPL with GATE18d)
# 5. ChemBERTa-77M and esm2_t6 and ankh_base (B6AEPL with GATE18d)
print(f"Protein Embeddings: {protein_embeddings}")
print(f"Ligand Embeddings: {ligand_embeddings}")

if len(ligand_embeddings) == 0: # 1. No embeddings
    model_arch = "GATE18e"
    stdict_paths = [
                "model/GATE18e_00AEPL_d0100_4_f0_best_stdict.pt",
                "model/GATE18e_00AEPL_d0100_4_f1_best_stdict.pt",
                "model/GATE18e_00AEPL_d0100_4_f2_best_stdict.pt",
                "model/GATE18e_00AEPL_d0100_4_f3_best_stdict.pt",
                "model/GATE18e_00AEPL_d0100_4_f4_best_stdict.pt"
                ]

elif len(protein_embeddings) == 0: # 2. ChemBERTa-77M only
    model_arch = "GATE18d"
    stdict_paths = [
                "model/GATE18d_00AEPL_d0100_f0_best_stdict.pt",
                "model/GATE18d_00AEPL_d0100_f1_best_stdict.pt",
                "model/GATE18d_00AEPL_d0100_f2_best_stdict.pt",
                "model/GATE18d_00AEPL_d0100_f3_best_stdict.pt",
                "model/GATE18d_00AEPL_d0100_f4_best_stdict.pt"
                ]
    
elif 'ankh_base' in protein_embeddings and 'esm2_t6' not in protein_embeddings:
    model_arch = "GATE18d"
    stdict_paths = [
                "model/GATE18d_B0AEPL_d0600_f0_best_stdict.pt",
                "model/GATE18d_B0AEPL_d0600_f1_best_stdict.pt",
                "model/GATE18d_B0AEPL_d0600_f2_best_stdict.pt",
                "model/GATE18d_B0AEPL_d0600_f3_best_stdict.pt",
                "model/GATE18d_B0AEPL_d0600_f4_best_stdict.pt"
                ]
        
elif 'esm2_t6' in protein_embeddings and 'ankh_base' not in protein_embeddings:
    model_arch = "GATE18d"
    stdict_paths = [
                "model/GATE18d_06AEPL_d0500_f0_best_stdict.pt",
                "model/GATE18d_06AEPL_d0500_f1_best_stdict.pt",
                "model/GATE18d_06AEPL_d0500_f2_best_stdict.pt",
                "model/GATE18d_06AEPL_d0500_f3_best_stdict.pt",
                "model/GATE18d_06AEPL_d0500_f4_best_stdict.pt"
                ]

elif 'ankh_base' in protein_embeddings and 'esm2_t6' in protein_embeddings:
    model_arch = "GATE18d"
    stdict_paths = [
                "model/GATE18d_B6AEPL_d0500_f0_best_stdict.pt",
                "model/GATE18d_B6AEPL_d0500_f1_best_stdict.pt",
                "model/GATE18d_B6AEPL_d0500_f2_best_stdict.pt",
                "model/GATE18d_B6AEPL_d0500_f3_best_stdict.pt",
                "model/GATE18d_B6AEPL_d0500_f4_best_stdict.pt"
                ]

else:
    raise ValueError("Invalid combination of protein and ligand embeddings found in dataset")
#------------------------------------------------------------------------------------------------------




print(f"Running Inference with {dataset_path} using model {model_arch}")
print(f"Model State Dict Paths:")
for path in stdict_paths: print(path)

# Loaders
test_loader = DataLoader(dataset = dataset, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)
print("Data Loader Created")

# Device Selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Ensemble Model
criterion = RMSELoss()
model_class = getattr(sys.modules[__name__], model_arch)
models = [model_class(
            dropout_prob=0, 
            in_channels=node_feat_dim,
            edge_dim=edge_feat_dim,
            conv_dropout_prob=0).float().to(device)
            for _ in range(len(stdict_paths))]


## MODEL NAME ##
model_paths = list(stdict_paths)
models = [load_model_state(model, path, device) for model, path in zip(models, model_paths)]

# Run inference
test_metrics = evaluate(models, test_loader, criterion, device, labels)



# Save the output, plot the results if there are labels in the dataset
#-------------------------------------------------------------------------------------------------------------------------
if labels:
    # Create a figure with a single plot (no subplots)
    fig, ax1 = plt.subplots(figsize=(8, 8))

    # Plot the predictions for test data only
    #loss, r, rmse, r2, y_true, y_pred = test_metrics
    loss, r, rmse, r2, y_true, y_pred, ids = test_metrics
    plot_predictions(ax1, y_true, y_pred, f"Predictions Inference\nR = {r:.3f}, RMSE = {rmse:.3f}", "Inference Predictions")

    # Save the y_true and y_pred in a single CSV file using the csv module
    with open(f'{dataset_path.split(".")[0]}_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'y_true', 'y_pred'])  # Write the header
        writer.writerows(sorted(zip(ids, y_true.tolist(), y_pred.tolist()), key=lambda x: x[0]))  # Write the data

    plt.tight_layout()
    plt.savefig(f'{dataset_path.split(".")[0]}_predictions.png', dpi=300)

else:
    y_true, y_pred, ids = test_metrics

    # Save the sorted y_true and y_pred in a single CSV file using the csv module
    with open(f'{dataset_path.split(".")[0]}_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'y_true', 'y_pred'])  # Write the header
        writer.writerows(sorted(zip(ids, y_true.tolist(), y_pred.tolist()), key=lambda x: x[0]))  # Write the sorted data
#-------------------------------------------------------------------------------------------------------------------------