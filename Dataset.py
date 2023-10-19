import os
import numpy as np
import torch
from torch_geometric.data import Dataset


class IG_Dataset(Dataset):
    def __init__(self, root):
        super().__init__(root)

        self.data_dir = root

        self.filepaths = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        self.input_data = {ind:torch.load(filepath) for ind, filepath in enumerate(self.filepaths)}

        # Transform labels to log space
        for ind in self.input_data:
            grph = self.input_data[ind]
            grph.affinity = torch.log(grph.affinity + 1)
            self.input_data[ind] = grph


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph