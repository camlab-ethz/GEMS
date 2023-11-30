import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class IG_Dataset(Dataset):
    def __init__(self, root, embedding=False, edge_features=False, atom_features=False, masternode=None, mn_self_loops=True):
        super().__init__(root)

        self.data_dir = root
        self.filepaths = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        self.input_data = {}
        
        # Process all the graphs according to kwargs
        ind = 0
        for file in self.filepaths[:20]:
            
            grph = torch.load(file)


            # Transform labels to negative log space and min-max scale
            min = 0
            max = 16
            pK = -torch.log10(grph.affinity)
            pK_scaled = (pK - min) / (max - min)


            # Generate feature matrix x with/without embedding and atom features
            if embedding:
                x = torch.cat((grph.x_lig, grph.x_prot_emb), axis=0)
            else:
                x = torch.cat((grph.x_lig, grph.x_prot_aa), axis=0)

            if not atom_features:
                x[:,-31:] = 0

            # If edge_features should be excluded, inclode only edge type and length
            if not edge_features:
                grph.edge_attr = grph.edge_attr[:,:4]
                #edge_attr_lig = grph.edge_attr_lig[:,:4]
                #edge_attr_prot = grph.edge_attr_prot[:,:4]


            # Number of nodes of the graph (including masternode)
            n_nodes = torch.tensor(x.shape[0], dtype=torch.int64)


            # Edge Index: If masternode, connect masternode to all atoms or only to ligand/protein atoms
            
            if masternode == None:
                
                train_graph = Data(x = x, 
                               edge_index=grph.edge_index, 
                               edge_attr=grph.edge_attr, 
                               y=pK_scaled
                               #,pos=grph.pos
                               )

            else:

                # If the masternode should be connected to all other nodes
                if masternode == 'all':
                    edge_index_master_prot_lig = torch.cat([grph.edge_index_master_prot, grph.edge_index_master_lig], axis = 1)
                    edge_index = torch.cat([grph.edge_index, edge_index_master_prot_lig], axis = 1)
                    if mn_self_loops:
                        mn_index = torch.max(edge_index).item()
                        edge_index = torch.cat((edge_index, torch.tensor([[mn_index],[mn_index]])), axis=1)
                        


                # If the masternode should only be connected to ligand nodes
                elif masternode == 'ligand':
                    edge_index = torch.cat([grph.edge_index, grph.edge_index_master_lig], axis = 1)
                    if mn_self_loops:
                        mn_index = torch.max(edge_index).item()
                        edge_index = torch.cat((edge_index, torch.tensor([[mn_index],[mn_index]])), axis=1)

                # If the masternode should only be connection to protein nodes
                elif masternode == 'protein':
                    edge_index = torch.cat([grph.edge_index, grph.edge_index_master_prot], axis = 1)
                    if mn_self_loops:
                        mn_index = torch.max(edge_index).item()
                        edge_index = torch.cat((edge_index, torch.tensor([[mn_index],[mn_index]])), axis=1)

                else: 
                    raise Exception("masternode must be either 'None', 'protein', 'ligand' or 'all'")
            
            
                train_graph = Data(x = x, 
                                edge_index=edge_index, 
                                edge_attr=grph.edge_attr, 
                                y=pK_scaled, 
                                n_nodes=n_nodes #needed for reading out masternode features
                                #,pos=grph.pos
                )         

            self.input_data[ind] = train_graph
            ind += 1


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph