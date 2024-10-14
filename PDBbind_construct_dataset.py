import os
import numpy as np
import importlib
import argparse
import torch
import json
import warnings
from torch_geometric.data import Dataset, Data
from Dataset import PDBbind_Dataset


def save_dataset(dataset, path):
    torch.save(dataset, path)


def main():

    parser = argparse.ArgumentParser()

    # Dataset construction
    parser.add_argument("--data_dir", required=True, help="The path to the folder containing all Data() (graphs) of the dataset.")
    parser.add_argument("--data_split", required=True, help="Filepath to dictionary (json file) containing the data split for the PDBbind dataset")
    parser.add_argument("--dataset", required=True, help="The dataset that should be loaded ['train', 'casf2013', 'casf2016']")
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the dataset')

    parser.add_argument("--refined_only", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If only the refined dataset should be used for training")
    parser.add_argument("--exclude_ic50", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints with IC50 affinity values should be excluded")
    parser.add_argument("--exclude_nmr", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints generated with NMR should be excluded")
    parser.add_argument("--resolution_threshold", default=5., type=float, help="Threshold for exclusion of datapoints with high resolution")
    parser.add_argument("--precision_strict", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints with unprecise affinity (>,<,..) should be excluded")

    # Graph Construction
    parser.add_argument('--protein_embeddings', nargs='+', default=[], help='Provide string to identify protein embeddings that should be incorporated (--protein embeddings string1 string2 string3).\
                        The strings should correspond to the keys that are used to save the embeddings in the graph object of the complexes'),
    parser.add_argument('--ligand_embeddings', nargs='+', default=[], help='Provide names of embeddings that should be incorporated (--ligand_embeddings string1 string2 string3).\
                        The strings should correspond to the keys that are used to save the embeddings in the graph object of the complexes'),
    parser.add_argument("--masternode", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If a masternode (mn) should be included in the graphs")
    parser.add_argument("--masternode_connectivity", default='all', help="If a mn is included, to which nodes it should be connected ('all', 'ligand', 'protein')")
    parser.add_argument("--masternode_edges", default='undirected', help='If the mn should be connected with undirected or directed edges ("undirected", "in", or "out")')
    parser.add_argument("--atom_features", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not Atom Features should be included")
    parser.add_argument("--edge_features", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not Edge Features should be included")
    parser.add_argument("--delete_ligand", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If ligand nodes should be deleted from the graph (ablation study)")
    parser.add_argument("--delete_protein", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If protein nodes should be deleted from the graph (ablation study)")

    args = parser.parse_args()

    protein_embeddings = args.protein_embeddings
    ligand_embeddings = args.ligand_embeddings

    dataset = PDBbind_Dataset(args.data_dir, args.dataset, args.data_split,
                        protein_embeddings, ligand_embeddings,
                        refined_only=args.refined_only,
                        exclude_ic50=args.exclude_ic50,
                        exclude_nmr=args.exclude_nmr,
                        resolution_threshold=args.resolution_threshold,
                        precision_strict=args.precision_strict,
                        delete_protein=args.delete_protein,
                        delete_ligand=args.delete_ligand,
                        masternode=args.masternode,
                        masternode_connectivity=args.masternode_connectivity,
                        masternode_edges=args.masternode_edges,
                        edge_features=args.edge_features, 
                        atom_features=args.atom_features)
    
    save_dataset(dataset, args.save_path)

if __name__ == "__main__":
    main()