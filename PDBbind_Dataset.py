import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class IG_Dataset(Dataset):
    def __init__(self,
                root,                               # Path to the folder containing the graphs  
                filepaths,                          # List of filepaths to the graphs that should be included
                embeddings,                         # List of all embeddings that should be included
                masternode=True,                    # If a masternode (mn) should be included
                masternode_connectivity = 'all',    # If a mn is included, to which nodes it should be connected ('all', 'ligand', 'protein')
                masternode_edges='undirected',      # If the mn should be connected with undirected or directed edges ("undirected", "in", or "out")
                edge_features=False,                # If edge features should be included
                atom_features=False,                # If atom features should be included
                refined_only=False,                 # If only refined complexes should be included
                exclude_ic50=False,                 # If IC50-labelled datapoints should be excluded
                exclude_nmr=False,                  # If NMR structures should be excluded
                resolution_threshold=5.,            # If structures with a resolution above this threshold should be excluded
                precision_strict=False,             # If only structures with affinity labels with '=' should be included
                delete_protein = False,             # If protein nodes should be deleted from the graph (ablation study)
                delete_ligand = False):             # If ligand nodes should be deleted from the graph (ablation study)
        
        super().__init__(root)

        self.data_dir = root
        self.filepaths = filepaths
        self.embeddings = embeddings
        self.PDBbind_data_dict = 

        with open('PDBbind_data_dict.json', 'r', encoding='utf-8') as json_file:
            self.pdbbind_dict = json.load(json_file)

        self.input_data = {}
        

        #------------------------------------------------------------------------------------------
        # Process all the graphs according to kwargs
        # -------------------------------------------------------------------------------------------
        ind = 0
        for file in self.filepaths:
            
            grph = torch.load(file)

            # GET METADATA [in_refined, affmetric_encoding[affinity_metric], resolution, precision_encoding[precision], float(log_kd_ki)]
            # NMR structures have resolution = 0
            # affmetric_encoding = {'Ki':1., 'Kd':2.,'IC50':3.}
            # precision_encoding = {'=':0., '>':1., '<':2., '>=':3., '<=':4., '~':5.}
            in_refined, affinity_metric, resolution, precision, log_kd_ki = grph.data.tolist()

            if refined_only and in_refined == 0:
                continue

            if resolution > resolution_threshold:
                #print(in_refined, affinity_metric, resolution, precision, log_kd_ki)
                continue

            if precision_strict and precision != 0:
                continue

            if exclude_ic50 and affinity_metric == 3:
                continue

            if exclude_nmr and resolution == 0:
                continue

            

            # Transform labels to negative log space and min-max scale
            min = 0
            max = 16
            pK = -torch.log10(grph.affinity)
            pK_scaled = (pK - min) / (max - min)

            #print(in_refined, affinity_metric, resolution, precision, log_kd_ki)


            # Generate feature matrix x with/without embedding and atom features
            if embedding:
                x = torch.cat((grph.x_lig_emb, grph.x_prot_emb), axis=0)
            else:
                x = torch.cat((grph.x_lig_aa, grph.x_prot_aa), axis=0)

            if not atom_features:
                x[:,-31:] = 0

            # If edge_features should be excluded, inclode only edge type and length
            if not edge_features:
                grph.edge_attr = grph.edge_attr[:,:4]
                #edge_attr_lig = grph.edge_attr_lig[:,:4]
                #edge_attr_prot = grph.edge_attr_prot[:,:4]


            # Number of nodes of the graph (including masternode)
            n_nodes = x.shape[0]
            n_lig_nodes = grph.x_lig_emb.shape[0]


            # Edge Index: If masternode, connect masternode to all atoms or only to ligand/protein atoms
            
            if masternode == 'None':
                
                train_graph = Data(x = x, 
                               edge_index=grph.edge_index, 
                               edge_attr=grph.edge_attr, 
                               y=pK_scaled
                               #,pos=grph.pos
                               #,id=grph.id
                               )

            else:
                mn_edges_feature_vector = torch.tensor([0., 1., 0.,                         # it's a mn connection
                                                        0,                                  # length is zero
                                                        0., 0., 0.,0.,0.,                   # bondtype = None
                                                        0.,                                 # is not conjugated
                                                        0.,                                 # is not in ring
                                                        0., 0., 0., 0., 0., 0.])            # No stereo
                
                
                # If the masternode should be connected to all other nodes
                if masternode == 'all':
                    edge_index_master_prot_lig = torch.cat([grph.edge_index_master_prot, grph.edge_index_master_lig], axis = 1)
                    edge_index = torch.cat([grph.edge_index, edge_index_master_prot_lig], axis = 1)
                    if mn_self_loops:
                        mn_index = torch.max(edge_index).item()
                        edge_index = torch.cat((edge_index, torch.tensor([[mn_index],[mn_index]])), axis=1)

                    mn_feature_rows = mn_edges_feature_vector.repeat(edge_index.shape[1] - grph.edge_attr.shape[0], 1)
                    edge_attr = torch.cat([grph.edge_attr, mn_feature_rows], axis=0)
                        

                # If the masternode should only be connected to ligand nodes
                elif masternode == 'ligand':
                    edge_index = torch.cat([grph.edge_index, grph.edge_index_master_lig], axis = 1)
                    if mn_self_loops:
                        mn_index = torch.max(edge_index).item()
                        edge_index = torch.cat((edge_index, torch.tensor([[mn_index],[mn_index]])), axis=1)
                    
                    mn_feature_rows = mn_edges_feature_vector.repeat(edge_index.shape[1] - grph.edge_attr.shape[0], 1)
                    edge_attr = torch.cat([grph.edge_attr, mn_feature_rows], axis=0)


                # If the masternode should only be connection to protein nodes
                elif masternode == 'protein':
                    edge_index = torch.cat([grph.edge_index, grph.edge_index_master_prot], axis = 1)
                    if mn_self_loops:
                        mn_index = torch.max(edge_index).item()
                        edge_index = torch.cat((edge_index, torch.tensor([[mn_index],[mn_index]])), axis=1)

                    mn_feature_rows = mn_edges_feature_vector.repeat(edge_index.shape[1] - grph.edge_attr.shape[0], 1)
                    edge_attr = torch.cat([grph.edge_attr, mn_feature_rows], axis=0)


                else: 
                    raise Exception("masternode must be either 'None', 'protein', 'ligand' or 'all'")
                



                if delete_protein: 

                    # Remove all nodes that don't belong to the ligand from feature matrix, keep masternode
                    x = torch.concat( [x[:n_lig_nodes,:] , x[-1,:].view(1,-1)] )

                    # Remove all coordinates of nodes that don't belong to the ligand and keep masternode (for visualization only)
                    grph.pos = torch.concat( [grph.pos[:n_lig_nodes,:] , grph.pos[-1,:].view(1,-1)] )

                    # Keep only edges that are between ligand atoms of between ligand atoms and masternode
                    edge_mask = ((edge_index < n_lig_nodes) | (edge_index == n_nodes-1)).all(dim=0)
                    edge_index = edge_index[:, edge_mask]
                    edge_index[edge_index == n_nodes-1] = n_lig_nodes
                    edge_attr = edge_attr[edge_mask, :]

                    n_nodes = x.shape[0]
            

                elif delete_ligand:

                    # Remove all nodes that don't belong to the ligand from feature matrix, keep masternode
                    x = x[n_lig_nodes:, :]

                    # Remove all coordinates of nodes that don't belong to the ligand and keep masternode (for visualization only)
                    grph.pos = grph.pos[n_lig_nodes:, :]

                    # Keep only edges that are between protein nodes and the masternode
                    edge_mask = torch.all(edge_index >= n_lig_nodes, dim=0)
                    edge_index = edge_index[:, edge_mask] - n_lig_nodes
                    edge_attr = edge_attr[edge_mask, :]

                    n_nodes = x.shape[0]


                train_graph = Data(x = x, 
                                edge_index=edge_index, 
                                edge_attr=edge_attr, 
                                y=pK_scaled, 
                                n_nodes=torch.tensor(n_nodes, dtype=torch.int64) #needed for reading out masternode features
                                #,pos=grph.pos
                                #,id=grph.id
                )



            self.input_data[ind] = train_graph
            ind += 1


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph
    



''' Do this in the Dataset Class
        #------------------------------------------------------------------------------------------
        # Retrieve the binding affinity and other metadata of the complex
        # -------------------------------------------------------------------------------------------
        affmetric_encoding = {'Ki':1., 'Kd':2.,'IC50':3.}
        precision_encoding = {'=':0., '>':1., '<':2., '>=':3., '<=':4., '~':5.}
    
        if 'Ki' in affinity_dict[complex_id].keys():
            affinity = affinity_dict[complex_id]['Ki']
            affinity_metric = 'Ki'
            
        elif 'Kd' in affinity_dict[complex_id].keys():
            affinity = affinity_dict[complex_id]['Kd']
            affinity_metric = 'Kd'

        elif 'IC50' in affinity_dict[complex_id].keys():
            affinity = affinity_dict[complex_id]['IC50']
            affinity_metric = 'IC50'

        resolution = affinity_dict[complex_id]['resolution']
        log_kd_ki = affinity_dict[complex_id]['log_kd_ki']
        precision = affinity_dict[complex_id]['precision']


        try: resolution = float(resolution)
        except ValueError: resolution = 0
        


        # Find out if the graph is part of the test or training data and save into the corresponding folder: 
        # -------------------------------------------------------------------------------------------
        log_string += 'Successful - Saved in '
        save_folders = []

        in_casf_2013 = False
        in_casf_2016 = False
        in_refined = False

        if complex_id in casf_2013_complexes:
            in_casf_2013 = True
            log_string += 'CASF2013 '
            save_folders.append(casf2013_folder)
            
        if complex_id in casf_2016_complexes:
            in_casf_2016 = True
            log_string += 'CASF2016 '
            save_folders.append(casf2016_folder)

        if (not in_casf_2013) and (not in_casf_2016):
            log_string += 'Training Data'
            save_folders.append(train_folder)
            in_refined = complex_id in refined_complexes
        #------------------------------------------------------------------------------------------ 
        
        metadata = [in_refined, affmetric_encoding[affinity_metric], resolution, precision_encoding[precision], float(log_kd_ki)]
        '''