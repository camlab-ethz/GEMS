import os
import numpy as np
import torch
from torch_geometric.data import Dataset


class IG_Dataset(Dataset):
    def __init__(self, root, embedding=False, edge_features=False, atom_features=False, masternode=None):
        super().__init__(root)

        self.data_dir = root
        self.embedding = embedding

        self.filepaths = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        
        self.input_data = {}
        
        # Process the input graphs
        ind = 0
        for file in self.filepaths:
            
            grph = torch.load(file)

            # Transform labels to negative log space and min-max scale
            min = 0
            max = 16
            pK = -torch.log10(grph.affinity)
            pK_scaled = (pK - min) / (max - min)
            grph.y = pK_scaled
            
            if self.embedding:
                grph.x = torch.cat((grph.x_lig, grph.x_prot_emb), axis=0)
            else:
                grph.x = torch.cat((grph.x_lig, grph.x_prot_aa), axis=0)

            if not edge_features:
                grph.edge_attr = grph.edge_attr[:,:3].view(-1,1)
                grph.edge_attr_lig = grph.edge_attr_lig[:,:3].view(-1,1)
                grph.edge_attr_prot = grph.edge_attr_prot[:,:3].view(-1,1)

            if not atom_features:
                grph.x[:, -31:] = 0

            grph.n_nodes = grph.x.shape[0]
            grph.data = grph.data.reshape(1, 3)

            
            # 
            if masternode == 'all':
            

            if masternode == 'ligand'

            if masternode == 'protein'

            # Merge the edge_index_master_lig and the edge_index_master_prot into a overall master node edge_index
            grph.edge_index_master_prot_lig = torch.cat([grph.edge_index_master_prot, grph.edge_index_master_lig], axis = 1)

            # Create an edge_index with all possible edges, convalent, non-covalent and directed master node connections
            grph.edge_index_master = torch.cat([grph.edge_index, grph.edge_index_master_prot_lig], axis = 1)
            
            
            self.input_data[ind] = grph
            ind += 1


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph


#     graph = Data(
        
            x_lig = torch.tensor(x_lig, dtype=torch.float64),
            x_prot_emb = torch.tensor(x_prot_emb, dtype=torch.float64),
            x_prot_aa = torch.tensor(x_prot_aa, dtype=torch.float64),
                 
            edge_index = edge_index,
            edge_index_lig = edge_index_lig,
            edge_index_prot = edge_index_prot,

            edge_index_master_lig = edge_index_master_lig,
            edge_index_master_prot = edge_index_master_prot,

            edge_attr = edge_attr,
            edge_attr_lig = edge_attr_lig,
            edge_attr_prot = edge_attr_prot,

            pos = torch.tensor(pos, dtype=torch.float64),
            affinity= torch.tensor(affinity, dtype=torch.float64),

            id = complex_id,
            data = torch.tensor(metadata, dtype=torch.float64)
            )
