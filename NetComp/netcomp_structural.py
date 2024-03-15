import os
import networkx as nx
import netcomp as nc

from scipy.sparse import csr_matrix, coo_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.loader import DataLoader
from Dataset_graph_similarity import IG_Dataset

from torch_geometric.utils import to_networkx, from_networkx



def compute_similarity(g1, g2):

    g1x = to_networkx(g1, node_attrs=['x'], edge_attrs=['edge_attr'], graph_attrs=['y', 'pos'], to_undirected=True)
    g2x = to_networkx(g2, node_attrs=['x'], edge_attrs=['edge_attr'], graph_attrs=['y', 'pos'], to_undirected=True)

    A1,A2 = [nx.adjacency_matrix(G) for G in [g1x,g2x]]

    return nc.lambda_dist(A1,A2,kind='laplacian',k=10)



def compute_pairwise_similarities(graphs):
    n = len(graphs)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):  # No need to calculate when j < i, matrix is symmetrical
            similarity = compute_similarity(graphs[i], graphs[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Fill in the symmetric value

    return similarity_matrix





# Paths to data
casf2016_dir = '/data/grbv/PDBbind/DTI_5_c3/test_data/'
casf2016_complexes = [filename[0:4] for filename in os.listdir(casf2016_dir) if 'graph' in filename]

# casf2013_dir = '/data/grbv/PDBbind/DTI_5/input_graphs_esm2_t6_8M/test_data/casf2013'
# casf2013_complexes = [filename[0:4] for filename in os.listdir(casf2013_dir) if 'graph' in filename]

# train_dir = '/data/grbv/PDBbind/DTI_5_c3/training_data'
# train_complexes = [filename[0:4] for filename in os.listdir(train_dir) if 'graph' in filename]


embedding = False
edge_features = False
atom_features = False
refined_only = False
exclude_ic50 = False
exclude_nmr = False
resolution_threshold = 3.0 # What dataset filtering has been applied?
precision_strict = False



test_dataset = IG_Dataset(casf2016_dir, embedding=embedding, edge_features=edge_features, atom_features=atom_features, 
                              refined_only=refined_only, exclude_ic50=exclude_ic50, exclude_nmr=exclude_nmr,
                              resolution_threshold=resolution_threshold, precision_strict=precision_strict)

# train_dataset = IG_Dataset(train_dir, embedding=embedding, edge_features=edge_features, atom_features=atom_features, 
#                               refined_only=refined_only, exclude_ic50=exclude_ic50, exclude_nmr=exclude_nmr,
#                               resolution_threshold=resolution_threshold, precision_strict=precision_strict)


graphs = [graph for graph in test_dataset]

pairwise_dist_matrix = compute_pairwise_similarities(graphs)

np.save('casf_2016_pairwise_dist.npy', pairwise_dist_matrix)


