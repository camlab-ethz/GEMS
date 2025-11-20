import argparse
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import spearmanr, kendalltau
from Dataset import *
from torch_geometric.loader import DataLoader
from model.GEMS18 import *


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
    id = []

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
            id.extend(graphbatch.id)

    # Calculate evaluation metrics
    eval_loss = total_loss / len(loader)

    # Pearson Correlation Coefficient
    corr_matrix = np.corrcoef(y_true, y_pred)
    r = corr_matrix[0, 1]

    # Link the predictions to the corresponding ids in a dictionary
    id_to_pred = dict(zip(id, zip(y_true, y_pred)))

    # R2 Score
    r2_score = 1 - np.sum((np.array(y_true) - np.array(y_pred)) ** 2) / np.sum((np.array(y_true) - np.mean(np.array(y_true))) ** 2)

    # RMSE in pK unit
    min=0
    max=16
    true_labels_unscaled = torch.tensor(y_true) * (max - min) + min
    predictions_unscaled = torch.tensor(y_pred) * (max - min) + min
    rmse = criterion(predictions_unscaled, true_labels_unscaled)
    return eval_loss, r, rmse, r2_score, true_labels_unscaled, predictions_unscaled, id_to_pred
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


def plot_predictions(y_true, y_pred, title, metrics='', filepath=None, axislim=14):
    plt.scatter(y_true, y_pred, alpha=0.5, c='blue')
    
    # Displaying the metrics in the top left corner of the plotting area
    plt.text(0.05, 0.95, metrics, fontsize=14, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='left')
    
    plt.plot([0, axislim], [0, axislim], color='red', linestyle='--')
    plt.xlabel('True pK Values', fontsize=12)
    plt.ylabel('Predicted pK Values', fontsize=12)
    plt.ylim(0, axislim)
    plt.yticks(fontsize=12)
    plt.xlim(0, axislim)
    plt.xticks(fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.savefig(filepath, dpi=300)
#-------------------------------------------------------------------------------------------------------------------------



def parse_args():
    parser = argparse.ArgumentParser(description="Testing Parameters and Input Dataset Control")

    # REQUIRED Arguments
    parser.add_argument("--stdicts", type=str, required=True, help="String of comma-separated paths to stdicts that should be tested as an ensemble")
    parser.add_argument("--dataset_path", required=True, help="The path to the test dataset pt file")

    # OPTIONAL Arguments 
    parser.add_argument("--model_arch", default="GEMS18d", help="The name of the model architecture")
    parser.add_argument("--save_path", default=None, help="The path where the results should be exported to")

    return parser.parse_args()



def main():
    args = parse_args()

    # Paths
    dataset_path = args.dataset_path
    stdicts = args.stdicts.split(',')
    save_path = args.save_path

    if save_path == None: save_path = os.path.dirname(dataset_path)

    # Load the datasets
    test_dataset = torch.load(dataset_path)
    test_loader = DataLoader(dataset = test_dataset, batch_size=128, shuffle=True, num_workers=4, persistent_workers=True)
    print(f'Dataset: {dataset_path}')

    # Emsemble Model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_arch = args.model_arch
    conv_dropout_prob = 0
    dropout_prob = 0
    criterion = RMSELoss()

    node_feat_dim = test_dataset[0].x.shape[1]
    edge_feat_dim = test_dataset[0].edge_attr.shape[1]

    model_class = getattr(sys.modules[__name__], model_arch)
    models = [model_class(
                dropout_prob=dropout_prob, 
                in_channels=node_feat_dim,
                edge_dim=edge_feat_dim,
                conv_dropout_prob=conv_dropout_prob).float().to(device)
                for _ in range(len(stdicts))]


    ## MODEL NAME ##
    model_paths = list(stdicts)
    #for m in model_paths: print(m)
    models = [load_model_state(model, path) for model, path in zip(models, model_paths)]
    print('Loaded models:')
    print(model_paths)


    # Run inference
    loss, r, rmse, r2_score, y_true, y_pred, id_to_pred = evaluate(models, test_loader, criterion, device)
    
    kendall_tau = kendalltau(y_true, y_pred)
    tau = kendall_tau.statistic
    
    spearman_rho = spearmanr(y_true, y_pred)
    rho = spearman_rho.statistic


    # Plotting
    #-------------------------------------------------------------------------------------------------------------------------
    test_dataset_name = os.path.basename(dataset_path).split('.')[0]

    # Save the predictions to a json file
    with open(os.path.join(save_path, f'{test_dataset_name}_predictions.json'), 'w', encoding='utf-8') as json_file:
        json.dump(id_to_pred, json_file, ensure_ascii=False, indent=4)

    # Save Predictions Scatterplot
    filepath = os.path.join(save_path, f'{test_dataset_name}_predictions.png')
    plot_predictions(y_true, y_pred, test_dataset_name, metrics=f"R = {r:.3f}\nR2 = {r2_score:.3f}\nkTau = {tau:.3f}\nspRho = {rho:.3f}", filepath=filepath, axislim=14)
    print(f'Predictions saved to {os.path.join(save_path, f"{test_dataset_name}_predictions")}')
    #-------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()