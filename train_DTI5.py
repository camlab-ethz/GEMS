import argparse
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import time

from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from Models_DTI5 import *
from Models_DTI6 import *


from Dataset_DTI5 import IG_Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Training Parameters and Input Dataset Control")
    parser.add_argument("--model", required=True, help="The name of the model architecture")
    parser.add_argument("--data_dir", required=True, help="The source path of the data e.g.")
    parser.add_argument("--log_path", required=True, help="The path for saving results and logs.")
    parser.add_argument("--loss_func", default='MSE', help="The loss function that will be used ['MSE', 'RMSE', 'wMSE', 'L1', 'Huber']")
    parser.add_argument("--optim", default='Adam', help="The optimizer that will be used ['Adam', 'Adagrad', 'SGD']")
    parser.add_argument("--wandb", default=True, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not the run should be streamed to Weights and Biases")
    parser.add_argument("--project_name", help="Project Name for the saving of run data to Weights and Biases")
    parser.add_argument("--run_name", required=True, help="Name of the Run to display in saved data and in Weights and Biases (string)")
    parser.add_argument("--n_folds", default=5, type=int, help="The number of stratified folds that should be generated (n-fold-CV)")
    parser.add_argument("--fold_to_train", default=0, type=int, help="Of the n_folds generated, on which fold should the model be trained")
    parser.add_argument("--num_epochs", default=1000, type=int, help="Number of Epochs the model should be trained (int)")
    parser.add_argument("--batch_size", default=256, type=int, help="The Batch Size that should be used for training (int)")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="The learning rate with which the model should train (float)")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="The weight decay parameter with which the model should train (float)")
    parser.add_argument("--conv_dropout", default=0, type=float, help="The dropout probability that should be applied in the convolutional layers")
    parser.add_argument("--dropout", default=0, type=float, help="The dropout probability that should be applied in the dropout layer")

    # Dataset construction
    parser.add_argument("--masternode", default='None', help="The type of master node connectivity ['None', 'protein', 'ligand', 'all', None]")
    parser.add_argument("--embedding", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not ESM embedding should be included")
    parser.add_argument("--atom_features", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not Atom Features should be included")
    parser.add_argument("--edge_features", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not Edge Features should be included")
    parser.add_argument("--refined_only", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If only the refined dataset should be used for training")
    parser.add_argument("--exclude_ic50", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints with IC50 affinity values should be excluded")
    parser.add_argument("--exclude_nmr", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints generated with NMR should be excluded")
    parser.add_argument("--resolution_threshold", default=5., type=float, help="Threshold for exclusion of datapoints with high resolution")
    parser.add_argument("--precision_strict", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints with unprecise affinity (>,<,..) should be excluded")
    parser.add_argument("--delete_ligand", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If dataset should be constructed omitting all ligand nodes")
    parser.add_argument("--delete_protein", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If dataset should be constructed omitting all protein nodes")

    # Early stopping
    parser.add_argument("--early_stopping",  default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If early stopping should be used to prevent overfitting")
    parser.add_argument("--early_stop_patience", default=100, type=int, help="For how many epochs the validation loss can cease to decrease without triggering early stop")
    parser.add_argument("--early_stop_min_delta", default=0.3, type=float, help="How far train loss and val loss are allowed to diverge without triggering early stop")



    # If the learning rate should be adaptive LINEAR
    parser.add_argument("--alr_lin",  default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Linear learning rate reduction scheme will be used")
    parser.add_argument("--start_factor", default=1, type=float,help="Factor by which the learning rate will be reduced. new_lr = lr * factor.")
    parser.add_argument("--end_factor", default=0.01, type=float, help="Factor by which the learning rate will be reduced in the last epoch. new_lr = lr * factor.")
    parser.add_argument("--total_iters", default=1000, type=float, help="The number of iterations after which the linear reduction of the LR should be finished")

    # If the learning rate should be adaptive MULTIPLICATIVE
    parser.add_argument("--alr_mult",  default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Multiplicative learning rate reduction scheme will be used")
    parser.add_argument("--factor", default=0.99, type=float, help="Factor by which the learning rate will be reduced. new_lr = lr * factor.")

    # If the learning rate should be adaptive REDUCEONPLATEAU
    parser.add_argument("--alr_plateau",  default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Adaptive learning rate reduction REDUCELRONPLATEAU scheme will be used")
    parser.add_argument("--reduction", default=0.1, type=float, help="Factor by which the LR should be reduced on plateau")
    parser.add_argument("--patience", default=10, type=int, help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--min_lr", default=0.5e-4, type=float, help="A lower bound on the learning rate")

    # If a state_dict should be loaded before the training
    parser.add_argument("--pretrained",  default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Provide the path of a state dict that should be imported")
    parser.add_argument("--start_epoch", default=0, type=int, help="Provide the starting epoch (in case of importing pretrained model)")

    return parser.parse_args()

args = parse_args()

torch.manual_seed(0)


# Training Parameters and Config
#----------------------------------------------------------------------------------------------------

# Architecture and run settings
model_arch = args.model
data_dir = args.data_dir
log_path = args.log_path
project_name = args.project_name
run_name = args.run_name
wandb_tracking = args.wandb

# Dataset Construction
embedding = args.embedding
atom_features = args.atom_features
edge_features = args.edge_features
masternode = args.masternode
refined_only = args.refined_only
exclude_ic50 = args.exclude_ic50
exclude_nmr = args.exclude_nmr
resolution_threshold = args.resolution_threshold
precision_strict = args.precision_strict
delete_ligand = args.delete_ligand
delete_protein = args.delete_protein

if wandb_tracking: print(f'Saving into Project Folder {project_name}')

random_seed = 42

# Training Parameters
loss_function = args.loss_func
num_epochs = args.num_epochs
learning_rate = args.learning_rate
weight_decay = args.weight_decay
optim = args.optim
batch_size = args.batch_size
dropout_prob = args.dropout
conv_dropout_prob = args.conv_dropout

n_folds = args.n_folds
fold_to_train = args.fold_to_train

save_dir = log_path
wandb_dir = log_path

run_name = f'{run_name}_f{fold_to_train}' 


# Early Stopping
early_stopping = args.early_stopping
if early_stopping:

    class EarlyStopper:
        def __init__(self, patience=1, min_delta=1):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_r = 0
            #self.min_validation_r = float('inf')

        def early_stop(self, val_r, train_r):
            if abs(val_r - train_r) > self.min_delta:
                print(f'Early Stopping: Difference Validation R - Train R has become higher than {self.min_delta} epochs')
                return True
            if val_r > self.min_validation_r:
                self.min_validation_r = val_r
                self.counter = 0
            elif val_r < self.min_validation_r:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f'Early Stopping: Validation R has not decreased for {self.patience} epochs')
                    return True
            return False
        
    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)



# Learning rate reduction scheme
alr_lin = args.alr_lin
alr_mult = args.alr_mult
alr_plateau = args.alr_plateau

alr = ''
if alr_lin:
    start_factor = args.start_factor
    end_factor = args.end_factor
    total_iters = args.total_iters
    alr += f'Linear: Start Factor {start_factor}, End Factor {end_factor}, Total Iters {total_iters}\n'

if alr_mult:
    factor = args.factor
    alr += f'Multiplicative: Factor {factor}\n'

if alr_plateau:
    patience = args.patience
    reduction = args.reduction
    min_lr = args.min_lr
    alr += f'Plateau: Patience {patience}, Reduction Factor {reduction}, Min LR {min_lr}'

if wandb_tracking: 
        config = {
                "Learning Rate": learning_rate,
                "ESM Embedding": embedding,
                "Edge Features": edge_features,
                "Atom Features": atom_features,
                "Masternode Connectivity": masternode,
                "Weight Decay": weight_decay,
                "Architecture": model_arch,
                "Epochs": num_epochs,
                "Optimizer": optim,
                "Early Stopping": early_stopping,
                "Early Stopping Patience": args.early_stop_patience,
                "Batch Size": batch_size,
                "Splitting Random Seed":random_seed,
                "Dropout Probability": dropout_prob,
                "Dropout Prob Convolutional Layers":conv_dropout_prob,
                "Adaptive LR Scheme": alr
                }

pretrained = args.pretrained
start_epoch = args.start_epoch

if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'Saving Directory generated')
#----------------------------------------------------------------------------------------------------





# Load Dataset - Split into training and validation set in a stratified way
#----------------------------------------------------------------------------------------------------

# Location of the training data
train_dir = data_dir

dataset = IG_Dataset(train_dir, 
                    embedding=embedding, 
                    edge_features=edge_features, 
                    atom_features=atom_features, 
                    masternode=masternode,
                    refined_only=refined_only,
                    exclude_ic50=exclude_ic50,
                    exclude_nmr=exclude_nmr,
                    resolution_threshold=resolution_threshold,
                    precision_strict=precision_strict, 
                    delete_ligand=delete_ligand,
                    delete_protein=delete_protein)
print(dataset)

node_feat_dim = dataset[0].x.shape[1]
edge_feat_dim = dataset[0].edge_attr.shape[1]

labels = [graph.y.item() for graph in dataset]
print(max(labels), len(labels))


# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)

group_assignment = np.array( [round(lab) for lab in labels] )

train_indices = []
val_indices = []
for i, (train_index, val_index) in enumerate(skf.split(np.zeros(len(dataset)), group_assignment)):
    val_indices.append(val_index.tolist())
    train_indices.append(train_index.tolist())


# Select the fold that should be used for the training
train_idx = train_indices[fold_to_train]
val_idx = val_indices[fold_to_train]

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

print(f'Length Training Dataset: {len(train_dataset)}')
print(f'Length Validation Dataset: {len(val_dataset)}')

train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
eval_loader_train = DataLoader(dataset = train_dataset, batch_size=512, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
eval_loader_val = DataLoader(dataset = val_dataset, batch_size=512, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
#----------------------------------------------------------------------------------------------------





# Plot the distributions of the datasets
#----------------------------------------------------------------------------------------------------
training_labels = [graph.y.item() for graph in train_dataset]
validation_labels = [graph.y.item() for graph in val_dataset]
highest_label = max([max(training_labels), max(validation_labels)])

def create_histogram(data, title, xlim, num_bins=50):

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12, 6))  # Set the figure size as needed

    # Create the histogram
    frequencies, bins, _ = plt.hist(data, bins=num_bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Labels')
    plt.ylabel('Count (Log Scale)')
    plt.title(title)

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Add labels to the columns
    for freq, bin_center, in zip(frequencies, bin_centers):
        plt.text(bin_center, freq+1, str(int(freq)), ha='center', rotation=90, va='bottom', fontweight='bold', fontsize=8)

    plt.yscale('linear')
    plt.xlim(0, np.ceil(xlim))

    return fig

hist_training_labels = create_histogram(training_labels, f'Labels Training Dataset', highest_label)
hist_validation_labels = create_histogram(validation_labels, f'Labels Validation Dataset', highest_label)
#----------------------------------------------------------------------------------------------------





# Initialize Model, Optimizer and Loss Function
#-------------------------------------------------------------------------------------------------------------------------------
 
# Function to count number of trainable parameters
def count_parameters(model, trainable=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


# # Device Settings
# num_threads = torch.get_num_threads() // 2
# torch.set_num_threads(num_threads)

# torch.cuda.set_device(device_idx)
# device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
# print(device, torch.cuda.current_device(), torch.cuda.get_device_name())

# Device Settings
num_threads = int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads()))
torch.set_num_threads(num_threads)

# Since SLURM sets CUDA_VISIBLE_DEVICES for us, the first available GPU will be "cuda:0" from this script's perspective.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device, torch.cuda.current_device(), torch.cuda.get_device_name())


# Initialize the model and optimizer
model_class = getattr(sys.modules[__name__], args.model)
Model = model_class(dropout_prob=dropout_prob, in_channels=node_feat_dim, edge_dim=edge_feat_dim, conv_dropout_prob=conv_dropout_prob).to(device)
Model = Model.double()
torch.save(Model, f'{save_dir}/model_configuration.pt')

parameters = count_parameters(Model)
print(f'Model architecture {model_arch} with {parameters} parameters')

if wandb_tracking: 
    config['Number of Parameters'] = parameters
    config['Device'] = torch.cuda.get_device_name()



if optim == 'Adam': optimizer = torch.optim.Adam(list(Model.parameters()),lr=learning_rate, weight_decay=weight_decay)
elif optim == 'Adagrad': optimizer = torch.optim.Adagrad(Model.parameters(), learning_rate, weight_decay=weight_decay)
elif optim == 'SGD': optimizer = torch.optim.SGD(Model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)



# Apply adaptive learning rate (alr) scheme
if alr_lin: 
    lin_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    learning_rate_reduction_scheme = f'Linear LR Scheduler enabled with start factor {start_factor}, end factor {end_factor} and total iters {total_iters}'
elif alr_mult:
    mult_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: factor)
    learning_rate_reduction_scheme = f'Multiplicative LR Scheduler enabled with factor {factor}'
elif alr_plateau: 
    plat_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=reduction, patience=patience, min_lr=min_lr)
    learning_rate_reduction_scheme = f'ReduceLRonPlateau LR Scheduler enabled with patience {patience}, factor {reduction} and min LR {min_lr}'
else: 
    learning_rate_reduction_scheme = 'No learning rate scheduler has been selected'

print(learning_rate_reduction_scheme)



# DEFINE LOSS FUNCTION ['MSE', 'wMSE', 'L1', 'Huber']

class wMSELoss(torch.nn.Module):
    def __init__(self):
        super(wMSELoss, self).__init__()
    def forward(self, output, targets):
        squared_errors = (output - targets) ** 2
        return torch.mean(squared_errors * (targets + 1))
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, output, targets):
        return torch.sqrt(self.mse(output, targets))


if loss_function == 'Huber':
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
    print(f'Loss Function: Huber Loss')

elif loss_function == 'L1':
    criterion = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    print(f'Loss Function: L1 Loss')

elif loss_function == 'wMSE':
    criterion = wMSELoss()
    print(f'Loss Function: wMSE Loss')

elif loss_function == 'RMSE':
    criterion = RMSELoss()
    print(f'Loss Function: RMSE Loss')

else: 
    criterion = torch.nn.MSELoss()
    print(f'Loss Function: MSE Loss')

#----------------------------------------------------------------------------------------------------



# Training Function for 1 Epoch
#-------------------------------------------------------------------------------------------------------------------------------
def train(Model, loader, criterion, optimizer, device):
    Model.train()
        
    # Initialize variables to accumulate metrics
    total_loss = 0.0
    y_true = []
    y_pred = []
                
    for graphbatch in loader:
        graphbatch.to(device)
        targets = graphbatch.y

        # Forward pass
        optimizer.zero_grad()
        output = Model(graphbatch).view(-1)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # Accumulate loss collect the true and predicted values for later use
        total_loss += loss.item()
        y_true.extend(targets.tolist())
        y_pred.extend(output.tolist())

    # Calculate evaluation metrics
    avg_loss = total_loss / len(loader)

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
    criter = RMSELoss()
    rmse = criter(predictions_unscaled, true_labels_unscaled)

    return avg_loss, r, rmse, r2_score, y_true, y_pred
#-------------------------------------------------------------------------------------------------------------------------------


# Evaluation Function
#-------------------------------------------------------------------------------------------------------------------------------
def evaluate(Model, loader, criterion, device):
    Model.eval()

    # Initialize variables to accumulate the evaluation results
    total_loss = 0.0
    y_true = []
    y_pred = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for graphbatch in loader:

            graphbatch.to(device)
            targets = graphbatch.y

            # Forward pass
            output = Model(graphbatch).view(-1)
            loss = criterion(output, targets)

            # Accumulate loss and collect the true and predicted values for later use
            total_loss += loss.item()
            y_true.extend(targets.tolist())
            y_pred.extend(output.tolist())


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
    criter = RMSELoss()
    rmse = criter(predictions_unscaled, true_labels_unscaled)

    return eval_loss, r, rmse, r2_score, y_true, y_pred
#-------------------------------------------------------------------------------------------------------------------------------








# Initialize WandB tracking with config dictionary
#-----------------------------------------------------------------------------------
if wandb_tracking:
    wandb.login()
    wandb.init(project=project_name, name = run_name, config=config, dir=wandb_dir)
    
    wandb.log({"Training Labels": wandb.Image(hist_training_labels),
               "Validation Labels": wandb.Image(hist_validation_labels)})


if pretrained:
    Model.load_state_dict(torch.load(pretrained))
    print(f'State Dict Loaded: {pretrained}')
    print(f'Start Epoch: {start_epoch}')
    epoch = start_epoch

else: 
    epoch = 0


print(f'Model Architecture {model_arch} - Fold {fold_to_train} ({run_name})')
print(f'Number of Parameters: {parameters}')
print(f'ESM Embedding: {embedding}')
print(f'Edge Features: {edge_features}')
print(f'Atom Features: {atom_features}')
print(f'Masternode Connectivity: {masternode}')

print(f'Learning Rate: {learning_rate}')
print(f'Weight Decay: {weight_decay}')
print(f'Batch Size: {batch_size}')
print(f'Loss Function: {loss_function}')    
print(f'Number of Epochs: {num_epochs}')
print(f'{learning_rate_reduction_scheme}\n')

print(f'Model Training Output ({run_name})')



# Plotting Functions
#-------------------------------------------------------------------------------------------------------------------------
def plot_predictions(train_y_true, train_y_pred, val_y_true, val_y_pred, title):
    
    axislim = 1.1
    fig = plt.figure(figsize=(8, 8))  # Set the figure size as needed

    plt.scatter(train_y_true, train_y_pred, alpha=0.5, c='blue', label='Training Data')
    plt.scatter(val_y_true, val_y_pred, alpha=0.5, c='red', label='Validation Data')

    plt.plot([min(train_y_true + val_y_true), axislim], [min(train_y_true + val_y_true), axislim], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.ylim(-0.1, axislim)
    plt.xlim(-0.1, axislim)
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.title(title)
    
    # Adding manual legend items for colors
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='Training Dataset'))
    legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=8, label='Validation Dataset'))

    plt.legend(handles=legend_elements, loc='upper left')
    return fig


def residuals_plot(train_y_true, train_y_pred, val_y_true, val_y_pred, title):
    axislim = 1.1
    fig = plt.figure(figsize=(8, 8))  # Set the figure size as needed

    plt.style.use('ggplot')
    train_residuals = np.array(train_y_true) - np.array(train_y_pred)
    val_residuals = np.array(val_y_true) - np.array(val_y_pred)

    # Plot training residuals in blue
    plt.scatter(train_y_pred, train_residuals, c='blue', label='Training Data', alpha=0.5)

    # Plot validation residuals in red
    plt.scatter(val_y_pred, val_residuals, c='red', label='Validation Data', alpha=0.5)

    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.xlim(-0.1, axislim)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(title)
    plt.legend()  # Add a legend to differentiate training and validation data
    plt.show()
    return fig
#-------------------------------------------------------------------------------------------------------------------------







# Training and Validation Set Performance BEFORE Training
#-------------------------------------------------------------------------------------------------------------------------------

train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred = evaluate(Model, eval_loader_train, criterion, device)
val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred = evaluate(Model, eval_loader_val, criterion, device)

log_string = f'Before Train: Train Loss: {train_loss:6.3f}|  Pearson:{train_r:6.3f}|  R2:{train_r2:6.3f}|  RMSE:{train_rmse:6.3f}|  -- Val Loss: {val_loss:6.3f}|  Pearson:{val_r:6.3f}|  R2:{val_r2:6.3f}|  RMSE:{val_rmse:6.3f}| '
print(log_string)

if wandb_tracking:
    wandb.log({
            "Epoch": epoch,
            "Learning Rate": optimizer.param_groups[0]['lr'],
            "Training Loss":train_loss,
            "Training Pearson Correlation": train_r,
            "Training RMSE": train_rmse,
            "Training R2": train_r2,
            "Validation Loss":val_loss,
            "Validation R2": val_r2,
            "Validation Pearson Correlation": val_r,
            "Validation RMSE": val_rmse
            })
#-------------------------------------------------------------------------------------------------------------------------------

# Initialize dictionary to store the current best epochs metrics
best_epoch = val_r
best_metrics = {'val': (val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred),
                'train': (train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred)}


plotted = []
last_saved_epoch = 0



#===============================================================================================================================================
# Training and Evaluation
#===============================================================================================================================================
tic = time.time()
for epoch in range(epoch+1, num_epochs+1):

    train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred = train(Model, train_loader, criterion, optimizer, device)


    # Validation Set Performance Between Training Epochs
    #-------------------------------------------------------------------------------------------------------------------------------    
    val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred = evaluate(Model, eval_loader_val, criterion, device)

    log_string = f'Epoch {epoch:05d}:  Train Loss: {train_loss:6.3f}|  Pearson:{train_r:6.3f}|  R2:{train_r2:6.3f}|  RMSE:{train_rmse:6.3f}|  -- Val Loss: {val_loss:6.3f}|  Pearson:{val_r:6.3f}|  R2:{val_r2:6.3f}|  RMSE:{val_rmse:6.3f}| '

    if wandb_tracking:
        wandb.log({
                "Epoch": epoch,
                "Learning Rate": optimizer.param_groups[0]['lr'],
                "Training Loss":train_loss,
                "Training Pearson Correlation": train_r,
                "Training RMSE": train_rmse,
                "Training R2": train_r2,
                "Validation Loss":val_loss,
                "Validation R2": val_r2,
                "Validation Pearson Correlation": val_r,
                "Validation RMSE": val_rmse
                })
        
    # Take a step in the activated learning rate reduction schemes
    if alr_plateau: plat_scheduler.step(val_loss)
    if alr_lin: lin_scheduler.step()
    if alr_mult: mult_scheduler.step()


    # If the previous best val_loss is beaten, save the model and update the best metrics dict
    if val_r > best_epoch: 
        torch.save(Model.state_dict(), f'{save_dir}/{run_name}_best_stdict.pt')
        log_string += ' Saved'
        last_saved_epoch = epoch

        best_epoch = val_r
        best_metrics['val'] = (val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred)
        best_metrics['train'] = (train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred)

    print(log_string, flush=True)

    if epoch % 50 == 0:
        print(f'Time: {((time.time() - tic)/60):5.0f}')



    early_stop = False
    if early_stopping: early_stop = early_stopper.early_stop(val_r, train_r)



    # After regular intervals, plot the predictions of the current and the best model
    # -------------------------------------------------------------------------------------------------------------------------------
    
    if epoch % 100 == 0 or epoch == num_epochs or early_stop:

        # Plot the predictions
        # predictions = plot_predictions( train_y_true, train_y_pred,
        #                                 val_y_true, val_y_pred,
        #                                 f"{run_name}: Epoch {epoch}\nTrain R = {train_r:.3f}, Validation R = {val_r:.3f}\n Train RMSE = {train_rmse:.3f}, Validation RMSE = {val_rmse:.3f}")


        # If there has been a new best epoch in the last interval of epochs, plot the predictions of this model      
        if last_saved_epoch not in plotted:
            
            # Load the current best metrics from dict
            val_loss, val_r, val_rmse, val_r2, val_y_true, val_y_pred = best_metrics['val']
            train_loss, train_r, train_rmse, train_r2, train_y_true, train_y_pred = best_metrics['train']

            # Plot the predictions and the residuals plot
            best_predictions = plot_predictions( train_y_true, train_y_pred,
                                    val_y_true, val_y_pred,
                                    f"{run_name}: Epoch {last_saved_epoch}\nTrain R = {train_r:.3f}, Validation R = {val_r:.3f}\nTrain RMSE = {train_rmse:.3f}, Validation RMSE = {val_rmse:.3f}")
            best_predictions.savefig(f'{save_dir}/predictions.png')

            residuals = residuals_plot(train_y_true, train_y_pred, val_y_true, val_y_pred, 
                                    f"{run_name}: Epoch {last_saved_epoch}\nTrain R = {train_r:.3f}, Validation R = {val_r:.3f}\nTrain RMSE = {train_rmse:.3f}, Validation RMSE = {val_rmse:.3f}")     
            residuals.savefig(f'{save_dir}/residuals.png')

            plotted.append(last_saved_epoch)

        plt.close('all')
        
        if wandb_tracking: 
            wandb.log({ #"Predictions Scatterplot": wandb.Image(predictions),
                        "Best Predictions Scatterplot": wandb.Image(best_predictions),
                        "Residuals Plot":wandb.Image(residuals)
                        })
            
    

    # Is it time for early stopping?
    if early_stop: break




toc = time.time()
training_time = (toc-tic)/60
print(f"Time for Training: {training_time:5.1f} minutes - ({(training_time/num_epochs):5.2f} minutes/epoch)")
if wandb_tracking: wandb.finish()