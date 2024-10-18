import os
import numpy as np
import torch
import json
from torch_geometric.data import Dataset, Data


class PDBbind_Dataset(Dataset): # RENAME THIS TO "DATASET" BEFORE PUBLIC RELEASE

    """
    A class used to represent a Dataset for protein-ligand interaction graphs. Takes as input a folder containing the graph.pth file for each graph that has been preprocessed 
    with the dataprep4_graph_construction.py script

    - Loads all graphs from a folder
    - If a data split dictionary is given, only the graphs that are included in the key that is provided in the dataset parameter are loaded.
    - If inference is set to True, labels are set to 0
    - If inference is set to False, a data dictionary containing the affinity labels for each complex has to be provided

    - For ablation studies, there are options to delete all protein_nodes from the graphs
    - To measure the contribution of atomic features and edge features, these can be excluded from the graph

    - A masternode can be included in the graph, which is connected to all nodes in the graph. The masternode can be connected to all nodes, only to ligand nodes or only to protein nodes.
    - The edges between the masternode and the other nodes can be directed from the ligand/protein nodes to the masternode, from the masternode to the ligand/protein nodes or undirected.
    - If a masternode is included, ablation tests can also include the deletion of ligand nodes from the graph.

    """



    def __init__(self,
                root,                               # Path to the folder containing the graphs

                # EMBEDDINGS TO BE INCLUDED
                protein_embeddings,                 # List of all protein embeddings that should be included (have to be saved in the graph Data() object already)
                ligand_embeddings,                  # List of all ligand embeddings that should be included (have to be saved in the graph Data() object already).

                # WHICH GRAPHS AND LABELS TO INCLUDE
                data_dict=None,                     # Path to dictionary containing the affinity labels of the complexes as dict[complex_id] = {'log_kd_ki': affinity}
                data_split=None,                    # Filepath to dictionary (json file) containing the data split for the graphs in the folder
                dataset=None,                       # If a split dict is given, which subset should be loaded ['train', 'test'] as defined in the data_split file

                # ABLATION
                delete_protein = False,             # If protein nodes should be deleted from the graph (ablation study)
                delete_ligand = False,              # If ligand nodes should be deleted from the graph (ablation study)
                edge_features=True,                 # If edge features should be included
                atom_features=True,                 # If atom features should be included

                # INCLUDE A MASTERNODE
                masternode=False,                   # If a masternode (mn) should be included
                masternode_connectivity = 'all',    # If a mn is included, to which nodes it should be connected ('all', 'ligand', 'protein')
                masternode_edges='undirected',      # If the mn should be connected with undirected or directed edges ("undirected", "in", or "out")
                ):                
                             
        
        super().__init__(root)

        # Data Directory and Embeddings to be included
        self.data_dir = root
        self.protein_embeddings = protein_embeddings
        self.ligand_embeddings = ligand_embeddings

        
        if data_dict is not None:
            with open(data_dict, 'r', encoding='utf-8') as json_file:
                self.data_dict = json.load(json_file)
            self.labels = True
        else:
            self.labels = False


        # Load the dictionary containing the data split for the dataset, if one is given
        # In no splitting dict is given, include all graphs in the folder in the dataset
        if data_split is not None:
            self.dataset = dataset
            self.data_split = data_split
            with open(self.data_split, 'r', encoding='utf-8') as json_file:
                self.split_dict = json.load(json_file) 

            # Generate the list of filepaths to the graphs that should be loaded
            included_complexes = self.split_dict[self.dataset]
            dataset_filepaths = [os.path.join(self.data_dir, f"{key}_graph.pth") for key in included_complexes]
            self.filepaths = [filepath for filepath in dataset_filepaths if os.path.isfile(filepath)]

        else: 
            self.filepaths = [file.path for file in os.scandir(self.data_dir) if file.name.endswith('graph.pth')]

        
        

        #------------------------------------------------------------------------------------------
        # Process all the graphs according to kwargs
        # -------------------------------------------------------------------------------------------
        self.input_data = {}
        ind = 0
        for file in self.filepaths:
            
            grph = torch.load(file)
            id = grph.id
            pos = grph.pos

            if self.labels:
                min=0
                max=16
                pK = self.data_dict[id]['log_kd_ki']
                pK_scaled = (pK - min) / (max - min)
            else: pK_scaled = 0


            # --- AMINO ACID EMBEDDINGS ---
            x = grph.x
            
            # Append the amino acid embeddings to the feature matrices
            for emb in self.protein_embeddings:
                emb_tensor = grph[emb]
                if emb_tensor is not None:
                    x = torch.concatenate((x, emb_tensor), axis=1)


            # --- LIGAND EMBEDDINGS ---
            ligand_embedding = None

            # Concatenate all ligand embeddings into a single vector
            for emb in self.ligand_embeddings:
                emb_vector = grph[emb]
                if emb_vector is None: print(f"Embedding {emb} not found for {id}")
                else:
                    if ligand_embedding is None: ligand_embedding = emb_vector
                    else:
                        ligand_embedding = torch.concatenate((ligand_embedding, emb_vector), axis=1)
                        ligand_embedding = ligand_embedding.float()


            # --- EDGE INDECES, EDGE ATTRIBUTES--- 
            # for convolution on 1) all edges, 2) only ligand edges and 3) only protein edges
            edge_index = grph.edge_index
            edge_index_lig = grph.edge_index_lig
            edge_index_prot = grph.edge_index_prot

            edge_attr = grph.edge_attr
            edge_attr_lig = grph.edge_attr_lig
            edge_attr_prot = grph.edge_attr_prot



            # If a masternode should be included in the graph, add the corresponding edge_index
            if masternode:

                # Depending on the desired masternode connectivity (to all nodes, to ligand nodes or to protein nodes),
                # choose the correct edge index master from the graph object
                if masternode_connectivity == 'all': edge_index_master = grph.edge_index_master
                elif masternode_connectivity == 'ligand': edge_index_master = grph.edge_index_master_lig
                elif masternode_connectivity == 'protein': edge_index_master = grph.edge_index_master_prot
                else: raise ValueError(f"Invalid value for masternode_connectivity: {masternode_connectivity}")

                # By default, the edge_index_master contains directed edges for information flow from the 
                # ligand and protein nodes to the masternode ("in")
                if masternode_edges == 'in':
                    pass

                # For information flow from the masternode to the ligand and protein nodes ("out"), swap the rows
                elif masternode_edges == 'out':
                    edge_index_master = edge_index_master[[1, 0], :]

                # For information flow in both directions ("undirected"), swap the rows and append
                elif masternode_edges == 'undirected':
                    edge_index_master = torch.concatenate((edge_index_master[:,:-1], edge_index_master[[1, 0], :]), dim=1)
                
                else: raise ValueError(f"Invalid value for masternode_edges: {masternode_edges}")

                # Append the updated edge_index_master to the edge_index containing the other edges in the graph
                edge_index = torch.concatenate((edge_index, edge_index_master), dim=1)
                edge_index_lig = torch.concatenate((edge_index_lig, edge_index_master), dim=1)
                edge_index_prot = torch.concatenate((edge_index_prot, edge_index_master), dim=1)


                # For each edge that has been added to connect the masternode, extend also the edge attribute 
                # matrix with a feature vector

                mn_edge_attr = torch.tensor([0., 1., 0.,        # it's a mn connection
                        0., 0., 0.,0.,                          # length is zero
                        0., 0., 0.,0.,0.,                       # bondtype = None
                        0.,                                     # is not conjugated
                        0.,                                     # is not in ring
                        0., 0., 0., 0., 0., 0.],                # No stereo
                        dtype=torch.float)

                mn_edge_matrix = mn_edge_attr.repeat(edge_index_master.shape[1], 1)

                edge_attr = torch.concatenate([edge_attr, mn_edge_matrix], axis=0)
                edge_attr_lig = torch.concatenate([edge_attr_lig, mn_edge_matrix], axis=0)
                edge_attr_prot = torch.concatenate([edge_attr_prot, mn_edge_matrix], axis=0)


            # If NO masternode should be included in the graph, remove the corresponding rows from pos and x
            else:
                x = x[:-1, :]
                pos = pos[:-1, :]



            # --- ABLATION STUDIES ---
            # For ablation studies: Generate feature matrices without node/edge features
            if not atom_features:
                x = torch.concatenate((x[:, 0:9], x[:, 40:]), dim=1)
            if not edge_features:
                edge_attr = edge_attr[:, :7]
                edge_attr_lig = edge_attr_lig[:, :7]
                edge_attr_prot = edge_attr_prot[:, :7]


            n_prot_nodes = grph.edge_index_master_prot.shape[1] - 1
            n_lig_nodes = grph.edge_index_master_lig.shape[1] - 1
            n_nodes = grph.edge_index_master.shape[1] - 1


            # For ablation studies: Generate graphs with all protein nodes removed          
            if delete_protein and delete_ligand: raise ValueError('Cannot delete both protein and ligand nodes')
            
            elif delete_protein and masternode:
                # Remove all nodes that don't belong to the ligand from feature matrix, keep masternode
                x = torch.concatenate( [x[:n_lig_nodes,:] , x[-1,:].view(1,-1)] )

                # Remove all coordinates of nodes that don't belong to the ligand and keep masternode
                pos = torch.concatenate( [pos[:n_lig_nodes,:] , pos[-1,:].view(1,-1)] )

                # Keep only edges that are between ligand atoms or between ligand atoms and masternode
                mask = ((edge_index < n_lig_nodes) | (edge_index == n_nodes)).all(dim=0)
                edge_index = edge_index[:, mask]
                edge_index[edge_index == n_nodes] = n_lig_nodes
                edge_attr = edge_attr[mask, :]

                train_graph = Data(x = x.float(),
                                edge_index=edge_index.long(),
                                edge_attr=edge_attr.float(),
                                y=torch.tensor(pK_scaled, dtype=torch.float),
                                n_nodes=torch.tensor([n_nodes, n_lig_nodes, n_prot_nodes], dtype=torch.long),
                                lig_emb=ligand_embedding
                                #,pos=pos
                                ,id=id
                                )
                
            elif delete_protein and not masternode:
                # Remove all nodes that don't belong to the ligand from feature matrix
                x = x[:n_lig_nodes,:]

                # Remove all coordinates of nodes that don't belong to the ligand
                pos = pos[:n_lig_nodes,:]

                # Keep only edges that are between ligand atoms
                mask = (edge_index < n_lig_nodes).all(dim=0)
                edge_index = edge_index[:, mask]
                edge_attr = edge_attr[mask, :]

                train_graph = Data(x = x.float(),
                                edge_index=edge_index.long(),
                                edge_attr=edge_attr.float(),
                                y=torch.tensor(pK_scaled, dtype=torch.float),
                                n_nodes=torch.tensor([n_nodes, n_lig_nodes, n_prot_nodes], dtype=torch.long),
                                lig_emb=ligand_embedding
                                #,pos=pos
                                ,id=id
                                )
                

            elif delete_ligand and not masternode: raise ValueError('Cannot delete ligand nodes without masternode')

            elif delete_ligand and masternode:
                # Remove all nodes that don't belong to the ligand from feature matrix, keep masternode
                x = x[n_lig_nodes:, :]

                # Remove all coordinates of nodes that don't belong to the ligand and keep masternode (for visualization only)
                grph.pos = grph.pos[n_lig_nodes:, :]

                # Keep only edges that are between protein nodes and the masternode
                mask = torch.all(edge_index >= n_lig_nodes, dim=0)
                edge_index = edge_index[:, mask] - n_lig_nodes
                edge_attr = edge_attr[mask, :]

                train_graph = Data(x = x.float(),
                                edge_index=edge_index.long(),
                                edge_attr=edge_attr.float(),
                                y=torch.tensor(pK_scaled, dtype=torch.float),
                                n_nodes=torch.tensor([n_nodes, n_lig_nodes, n_prot_nodes], dtype=torch.long),
                                lig_emb=ligand_embedding
                                #,pos=pos
                                ,id=id
                                )




            # --- NO ABLATION - Complete Interaction Graphs ---
            else: 
                train_graph = Data(x = x.float(), 
                                edge_index=edge_index.long(),
                                edge_attr=edge_attr.float(),
                                # To do: If we want to do convolution on only ligand or protein edges, 
                                # we need to pass the corresponding edge_index conaining these edges, but in such 
                                # architectures we can't do the ablation anymore because there we don't have
                                # any edge_index_lig or edge_index_prot.
                                #edge_index_lig=edge_index_lig,
                                #edge_index_prot=edge_index_prot,
                                #edge_attr_lig=edge_attr_lig,
                                #edge_attr_prot=edge_attr_prot,
                                y=torch.tensor(pK_scaled, dtype=torch.float),
                                n_nodes=torch.tensor([n_nodes, n_lig_nodes, n_prot_nodes], dtype=torch.long),
                                lig_emb=ligand_embedding
                                #,pos=pos
                                ,id=id
                                )


            self.input_data[ind] = train_graph
            ind += 1


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph