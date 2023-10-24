import os
import numpy as np
import torch
from torch_geometric.data import Dataset


class IG_Dataset(Dataset):
    def __init__(self, root, embedding=False, edge_features=False):
        super().__init__(root)

        self.data_dir = root
        self.embedding = embedding

        self.filepaths = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        self.input_data = {ind:torch.load(filepath) for ind, filepath in enumerate(self.filepaths)}

        # Process the input graphs
        min = 2
        max = 30
        for ind in self.input_data:
            grph = self.input_data[ind]

            # Transform labels to negative log space, clip and min-max scale
            log = torch.clamp(-torch.log(grph.affinity), min, max)
            grph.affinity = (log - min) / (max - min)
            
            if self.embedding:
                grph.x = torch.cat((grph.x_lig, grph.x_prot_emb), axis=0)
            else:
                grph.x = torch.cat((grph.x_lig, grph.x_prot_aa), axis=0)

            if not edge_features:
                grph.edge_attr = grph.edge_attr[:,3].view(-1,1)
                grph.edge_attr_lig = grph.edge_attr_lig[:,3].view(-1,1)
                grph.edge_attr_prot = grph.edge_attr_prot[:,3].view(-1,1)

            self.input_data[ind] = grph


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph