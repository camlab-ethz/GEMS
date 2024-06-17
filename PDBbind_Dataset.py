import os
import numpy as np
import argparse
import torch
import json
import warnings
from torch_geometric.data import Dataset, Data


class IG_Dataset(Dataset):
    def __init__(self,
                # Dataset Construction
                root,                               # Path to the folder containing the graphs  
                dataset,                            # Wheter the training or test data should be loaded ['train', 'casf2013', 'casf2016']
                data_split,                         # Filepath to dictionary (json file) containing the data split for the PDBbind dataset
                protein_embeddings,                 # List of all protein embeddings that should be included
                ligand_embeddings,                  # List of all ligand embeddings that should be included
                refined_only=False,                 # If only refined complexes should be included
                exclude_ic50=False,                 # If IC50-labelled datapoints should be excluded
                exclude_nmr=False,                  # If NMR structures should be excluded
                resolution_threshold=5.,            # If structures with a resolution above this threshold should be excluded
                precision_strict=False,             # If only structures with affinity labels with '=' should be included
                delete_protein = False,             # If protein nodes should be deleted from the graph (ablation study)
                delete_ligand = False,              # If ligand nodes should be deleted from the graph (ablation study)
                masternode=True,                    # If a masternode (mn) should be included
                masternode_connectivity = 'all',    # If a mn is included, to which nodes it should be connected ('all', 'ligand', 'protein')
                masternode_edges='undirected',      # If the mn should be connected with undirected or directed edges ("undirected", "in", or "out")
                edge_features=True,                 # If edge features should be included
                atom_features=True):                # If atom features should be included
                             
        
        super().__init__(root)

        self.data_dir = root
        self.dataset = dataset
        self.data_split = data_split
        self.protein_embeddings = protein_embeddings
        self.ligand_embeddings = ligand_embeddings
        if len(self.ligand_embeddings) > 0 and not masternode: warnings.warn("Ligand embeddings are not used if no masternode is included.")

        # Load the PDBbind data dictionary containing all metadata for the complexes
        self.PDBbind_data_dict = 'PDBbind_data_dict.json'
        with open('PDBbind_data_dict.json', 'r', encoding='utf-8') as json_file:
            self.pdbbind_dict = json.load(json_file)

        # Load the dictionary containing the data split for the PDBbind dataset
        with open(self.data_split, 'r', encoding='utf-8') as json_file:
            self.pdbbind_data_split = json.load(json_file) 


        # Generate the list of filepaths to the graphs that should be loaded
        pdbbind_data_split = self.pdbbind_data_split[self.dataset]
        dataset_filepaths = [os.path.join(self.data_dir, f"{key}_graph.pth") for key in pdbbind_data_split]
        self.filepaths = [filepath for filepath in dataset_filepaths if os.path.isfile(filepath)]

        
        

        #------------------------------------------------------------------------------------------
        # Process all the graphs according to kwargs
        # -------------------------------------------------------------------------------------------
        self.input_data = {}
        ind = 0
        for file in self.filepaths:
            
            grph = torch.load(file)
            id = grph.id
            pos = grph.pos
            print(id)

            # --- FILTERING ---
            if precision_strict and self.pdbbind_dict[id]['precision'] != "=": continue
            if exclude_ic50 and "IC50" in self.pdbbind_dict[id].keys(): continue
            if refined_only and 'refined' not in self.pdbbind_dict[id]['dataset']: continue
            if exclude_nmr and self.pdbbind_dict[id]['resolution'] == 'NMR': continue
            try: 
                if float(self.pdbbind_dict[id]['resolution']) > resolution_threshold: continue
            except ValueError: pass # If the resolution cannot be converted to string, it is an NMR structure

            
            # Get the affinity label and normalize it
            min=0
            max=16
            pK = self.pdbbind_dict[id]['log_kd_ki']
            pK_scaled = (pK - min) / (max - min)


            # --- NODE FEATURES ---
            x = grph.x
            
            # Append the amino acid embeddings to the feature matrices
            for emb in self.protein_embeddings:
                emb_tensor = grph[emb]
                if emb_tensor is not None:
                    x = torch.concatenate((x, emb_tensor), axis=1)


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

                # Append the ligand embeddings to the feature matrices 
                # (only to the masternode, which is the last row in the feature matrix)
                for emb in self.ligand_embeddings:
                    emb_vector = grph[emb]
                    if emb_tensor is not None:
                        emb_tensor = torch.concatenate((torch.zeros(x.shape[0]-1, emb_vector.shape[1]), emb_vector), axis=0)
                        x = torch.concatenate((x, emb_tensor), axis=1)

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
                        0., 0., 0., 0., 0., 0.])                # No stereo

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
            elif delete_protein and not masternode: raise ValueError('Cannot delete protein nodes without masternode')
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

                train_graph = Data(x = x,
                                edge_index=edge_index,
                                edge_attr=edge_attr, 
                                y=pK_scaled, 
                                n_nodes=torch.tensor([n_nodes, n_lig_nodes, n_prot_nodes], dtype=torch.int64) #needed for reading out masternode features
                                #,pos=pos
                                #,id=id
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

                train_graph = Data(x = x,
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                y=pK_scaled, 
                                n_nodes=torch.tensor(n_nodes, dtype=torch.int64) #needed for reading out masternode features
                                #,pos=pos
                                #,id=id
                )

            # --- 
            else: 
                train_graph = Data(x = x, 
                                edge_index=edge_index,
                                edge_attr=edge_attr, 
                                # To do: If we want to do convolution on only ligand or protein edges, 
                                # we need to pass the corresponding edge_index conaining these edges, but in such 
                                # architectures we can't do the ablation anymore because there we don't have
                                # any edge_index_lig or edge_index_prot.
                                #edge_index_lig=edge_index_lig,
                                #edge_index_prot=edge_index_prot,
                                #edge_attr_lig=edge_attr_lig,
                                #edge_attr_prot=edge_attr_prot,
                                y=pK_scaled,
                                n_nodes=torch.tensor(n_nodes, dtype=torch.int64) #needed for reading out masternode features
                                #,pos=pos
                                #,id=id
                )


            #print(train_graph)
            self.input_data[ind] = train_graph
            ind += 1


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph




def save_dataset(dataset, path):
    torch.save(dataset, path)



def main():

    parser = argparse.ArgumentParser()

    # Dataset construction
    parser.add_argument("--data_dir", required=True, help="The source path of the data e.g.")
    parser.add_argument("--data_split", required=True, help="Path to a dictionary (json) that contains the split of the data into training and test set")
    parser.add_argument("--dataset", required=True, help="The dataset that should be loaded ['train', 'casf2013', 'casf2016']")
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the dataset')

    parser.add_argument("--refined_only", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If only the refined dataset should be used for training")
    parser.add_argument("--exclude_ic50", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints with IC50 affinity values should be excluded")
    parser.add_argument("--exclude_nmr", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints generated with NMR should be excluded")
    parser.add_argument("--resolution_threshold", default=5., type=float, help="Threshold for exclusion of datapoints with high resolution")
    parser.add_argument("--precision_strict", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If datapoints with unprecise affinity (>,<,..) should be excluded")

    # Graph Construction
    #parser.add_argument("--protein_embeddings", default=[], type=list, help="List of all ligand embeddings that should be included")
    parser.add_argument('--protein_embeddings', nargs='+', help='Provide string to identify protein embeddings that should be incorporated (--protein embeddings string1 string2 string3).\
                        The strings should correspond to the keys that are used to save the embeddings in the graph object of the complexes'),
    parser.add_argument('--ligand_embeddings', nargs='+', help='Provide names of embeddings that should be incorporated (--ligand_embeddings string1 string2 string3).\
                        The strings should correspond to the keys that are used to save the embeddings in the graph object of the complexes'),
    # parser.add_argument("--ligand_embeddings", default=[], type=list, help="List of all ligand embeddings that should be included")
    parser.add_argument("--masternode", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If a masternode (mn) should be included in the graphs")
    parser.add_argument("--masternode_connectivity", default='all', help="If a mn is included, to which nodes it should be connected ('all', 'ligand', 'protein')")
    parser.add_argument("--masternode_edges", default='undirected', help='If the mn should be connected with undirected or directed edges ("undirected", "in", or "out")')
    parser.add_argument("--atom_features", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not Atom Features should be included")
    parser.add_argument("--edge_features", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not Edge Features should be included")
    parser.add_argument("--delete_ligand", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If ligand nodes should be deleted from the graph (ablation study)")
    parser.add_argument("--delete_protein", default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If protein nodes should be deleted from the graph (ablation study)")


    # Add additional arguments as needed
    args = parser.parse_args()

    protein_embeddings = args.protein_embeddings
    ligand_embeddings = args.ligand_embeddings

    dataset = IG_Dataset(args.data_dir, args.dataset, args.data_split,
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