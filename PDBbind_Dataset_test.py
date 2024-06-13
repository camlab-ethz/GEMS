import argparse
import sys
import os
import torch
#import matplotlib.pyplot as plt
# import numpy as np
# import wandb
# import time

# from torch_geometric.loader import DataLoader
# from sklearn.model_selection import StratifiedKFold
# from torch.utils.data import Subset
# from models_masternode import *
# from models_global_pool import *


from PDBbind_Dataset import IG_Dataset


# Dataset Construction
train_dir = 'PDBbind1'
filepaths = [grph.path for grph in os.scandir(train_dir) if grph.name.endswith('graph.pth')]
protein_embeddings = ['ankh', 'esm2_t6']
ligand_embeddings = ['ChemBERTa_10M']
atom_features = True
edge_features = True
masternode = True
refined_only = False
exclude_ic50 = False
exclude_nmr = False
resolution_threshold = 5
precision_strict = False
delete_ligand = False
delete_protein = False

dataset = IG_Dataset(   train_dir,
                        filepaths=filepaths,
                        protein_embeddings=protein_embeddings,
                        ligand_embeddings=ligand_embeddings,
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