import os
import numpy as np
import importlib
import argparse
import torch
import json
import warnings
from torch_geometric.data import Dataset, Data
from Dataset import PDBbind_Dataset


"""
This script constructs a dataset from the given data directory containing all Data() objects (graphs) of the dataset.
For explanation of command-line arguments, see argparse below.

Example Usage:
    python construct_dataset.py --data_dir /path/to/data --protein_embeddings ankh_base esm2_t6 --ligand_embeddings ChemBERTa_77M ChemBERTa_10M
    --data_dict /path/to/data_dict --data_split /path/to/data_split --dataset train --save_path /path/to/save.pt
"""


def save_dataset(dataset, path):
    torch.save(dataset, path)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="The path to the folder containing all Data() objects (graphs) of the dataset.")    
    parser.add_argument('--save_path', required=True, type=str, help='Path to save the dataset ending with .pt')
    
    # EMBEDDINGS TO BE INCLUDED
    parser.add_argument('--protein_embeddings', nargs='+', default=[], help='Provide string to identify protein embeddings that should be incorporated (--protein embeddings string1 string2 string3).\
                        The strings should correspond to the keys that are used to save the embeddings in the graph object of the complexes'),
    parser.add_argument('--ligand_embeddings', nargs='+', default=[], help='Provide names of embeddings that should be incorporated (--ligand_embeddings string1 string2 string3).\
                        The strings should correspond to the keys that are used to save the embeddings in the graph object of the complexes'),
    
    # WHICH GRAPHS AND LABELS TO INCLUDE
    parser.add_argument("--data_dict", default=None,, help="Path to dictionary containing the affinity labels of the complexes as dict[complex_id] = {'log_kd_ki': affinity}")
    parser.add_argument("--data_split", default=None,, help="Filepath to dictionary (json file) containing the data split for the graphs in the folder")
    parser.add_argument("--dataset", default=None,, help="If a split dict is given, which subset should be loaded ['train', 'test'] as defined in the data_split file")
    
    # ABLATION
    parser.add_argument("--atom_features", default=True, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not Atom Features should be included")
    parser.add_argument("--edge_features", default=True, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not Edge Features should be included")
    parser.add_argument("--delete_ligand", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If ligand nodes should be deleted from the graph (ablation study)")
    parser.add_argument("--delete_protein", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If protein nodes should be deleted from the graph (ablation study, only if masternode included")
    
    # INCLUDE A MASTERNODE
    parser.add_argument("--masternode", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If a masternode (mn) should be included in the graphs")
    parser.add_argument("--masternode_connectivity", default='all', help="If a mn is included, to which nodes it should be connected ('all', 'ligand', 'protein')")
    parser.add_argument("--masternode_edges", default='undirected', help='If the mn should be connected with undirected or directed edges ("undirected", "in", or "out")')
 


    args = parser.parse_args()

    dataset = PDBbind_Dataset(
                    args.data_dir,

                    # EMBEDDINGS TO BE INCLUDED
                    protein_embeddings=args.protein_embeddings,
                    ligand_embeddings=args.ligand_embeddings,

                    # WHICH GRAPHS AND LABELS TO INCLUDE
                    data_dict=args.data_dict,
                    data_split=args.data_split,
                    dataset=args.dataset, 
                    
                    # ABLATION
                    delete_protein=args.delete_protein,
                    delete_ligand=args.delete_ligand,
                    edge_features=args.edge_features, 
                    atom_features=args.atom_features,

                    # INCLUDE A MASTERNODE
                    masternode=args.masternode,
                    masternode_connectivity=args.masternode_connectivity,
                    masternode_edges=args.masternode_edges
    )
    
    save_dataset(dataset, args.save_path)

if __name__ == "__main__":
    main()