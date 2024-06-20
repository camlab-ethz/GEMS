import argparse
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb

from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from models_masternode import *
from models_global_pool import *


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, output, targets):
        return torch.sqrt(self.mse(output, targets))


def load_model_state(model, state_dict_path):
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()  # Set the model to evaluation mode
    return model



# Evaluation Function
#-------------------------------------------------------------------------------------------------------------------------------
def evaluate(models, loader, criterion, device):
    
    # Initialize variables to accumulate the evaluation results
    total_loss = 0.0
    y_true = []
    y_pred = []
    #id = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for graphbatch in loader:
            graphbatch.to(device)
            targets = graphbatch.y

            # Forward pass EMSEMBLE MODEL
            outputs = []
            for model in models: 
                outputs.append(model(graphbatch).view(-1))
            output = torch.mean(torch.stack(outputs), dim=0)    
            loss = criterion(output, targets)

            # Accumulate loss and collect the true and predicted values for later use
            total_loss += loss.item()
            y_true.extend(targets.tolist())
            y_pred.extend(output.tolist())
            #id.extend(graphbatch.id)

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
    return eval_loss, r, rmse, r2_score, true_labels_unscaled, predictions_unscaled#, id
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
    parser.add_argument("--model_arch", required=True, help="The name of the model architecture")
    parser.add_argument("--model_name", required=True, help="The name of the run")

    # Location of the test data 
    parser.add_argument("--casf2013_dataset_path", required=True, help="The path to the casf2014 test dataset pt file")
    parser.add_argument("--casf2016_dataset_path", required=True, help="The path to the casf2016 test dataset pt file")
    parser.add_argument("--train_dataset_path", required=True, help="The path to the training dataset pt file")

    parser.add_argument("--stdict_paths", type=str, required=True, help="String of comma-separated paths to stdicts that should be tested as an ensemble")
    parser.add_argument("--save_path", required=True, help="The path where the results should be exported to")

    return parser.parse_args()

args = parse_args()


# Paths
casf2013_dataset_path = args.casf2013_dataset_path
casf2016_dataset_path = args.casf2016_dataset_path
train_dataset_path = args.train_dataset_path
stdict_paths = args.stdict_paths.split(',')
save_path = args.save_path


# Load the datasets
casf2013_dataset = torch.load(casf2013_dataset_path)
casf2016_dataset = torch.load(casf2016_dataset_path)
train_dataset = torch.load(train_dataset_path)


node_feat_dim = casf2013_dataset[0].x.shape[1]
edge_feat_dim = casf2013_dataset[0].edge_attr.shape[1]
N = len(train_dataset)


# Loaders
casf2013_loader = DataLoader(dataset = casf2013_dataset, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)
casf2016_loader = DataLoader(dataset = casf2016_dataset, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)
train_loader = DataLoader(dataset = train_dataset, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)


# Title of the plots that will be generated
plot_titles = os.path.basename(train_dataset_path)



# Emsemble Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_arch = args.model_arch
conv_dropout_prob = 0
dropout_prob = 0
criterion = RMSELoss()

model_class = getattr(sys.modules[__name__], model_arch)
models = [model_class(
            dropout_prob=dropout_prob, 
            in_channels=node_feat_dim,
            edge_dim=edge_feat_dim,
            conv_dropout_prob=conv_dropout_prob).float().to(device)
            for _ in range(len(stdict_paths))]


## MODEL NAME ##
model_name = args.model_name
model_paths = list(stdict_paths)
#model_paths = [os.path.join(state_dict_path, file) for file in os.listdir(state_dict_path) if file.startswith(model_name)]
for m in model_paths: print(m)
models = [load_model_state(model, path) for model, path in zip(models, model_paths)]




# Performance on the complete test datasets
#-------------------------------------------------------------------------------------------------------------------------
casf2016_metrics = evaluate(models, casf2016_loader, criterion, device)
casf2013_metrics = evaluate(models, casf2013_loader, criterion, device)
train_metrics = evaluate(models, train_loader, criterion, device)

# Create a figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

# Plot the predictions for CASF2016
loss, r, rmse, r2, y_true, y_pred = casf2016_metrics #,id    
plot_predictions(ax1, y_true, y_pred, f"CASF2016 Predictions\nR = {r:.3f}, RMSE = {rmse:.3f}", 'CASF2016 Benchmark Data')

# Plot the predictions for CASF2013
loss, r, rmse, r2, y_true, y_pred = casf2013_metrics #,id
plot_predictions(ax2, y_true, y_pred, f"CASF2013 Predictions\nR = {r:.3f}, RMSE = {rmse:.3f}", 'CASF2013 Benchmark Data')

# Plot the predictions for the training data
loss, r, rmse, r2, y_true, y_pred = train_metrics #,id
plot_predictions(ax3, y_true, y_pred, f"Training Predictions\nR = {r:.3f}, RMSE = {rmse:.3f}", 'Training Data')

plt.tight_layout()
plt.savefig(f'{save_path}/test_scatterplot.png', dpi=300)
#-------------------------------------------------------------------------------------------------------------------------