import os
import numpy as np
import torch
from torch_geometric.data import Dataset
import argparse
from torch_geometric.loader import DataLoader
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Training Parameters and Input Dataset Control")
    parser.add_argument("--data_dir", required=True, help="The source path of the data")
    return parser.parse_args()

args = parse_args()



class IG_Dataset(Dataset):
    def __init__(self, root, embedding=False, edge_features=False, atom_features=False):
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
                grph.edge_attr = grph.edge_attr[:,3].view(-1,1)
                grph.edge_attr_lig = grph.edge_attr_lig[:,3].view(-1,1)
                grph.edge_attr_prot = grph.edge_attr_prot[:,3].view(-1,1)

            if not atom_features:
                grph.x[:, -31:] = 0


            grph.n_nodes = grph.x.shape[0]
            grph.data = grph.data.reshape(1, 3)

            
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
    


data_dir = args.data_dir



def benchmark_two_versions (num_workers, persistent_workers, pin_memory):

    dataset = IG_Dataset(data_dir)
    train_loader = DataLoader(dataset = dataset, batch_size=256, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)

    tic = time.time()
    for epoch in range(10):
        for graphbatch in train_loader:
            pass
    toc = time.time()
    print(f'--- Time for 10 epochs: {toc-tic}')


tests = [   ('Base Case', 0, False, False),

            ('Num workers 2', 2, False, False),
            ('Num workers 2 persistent', 2, True, False),
            ('Num workers 2 persistent pin', 2, True, True),

            ('Num workers 4', 4, False, False),
            ('Num workers 4 persistent', 4, True, False),
            ('Num workers 4 persistent pin', 4, True, True),

            ('Num workers 6', 6, False, False),
            ('Num workers 6 persistent', 6, True, False),
            ('Num workers 6 persistent pin', 6, True, True),

            ('Num workers 8', 8, False, False),
            ('Num workers 8 persistent', 8, True, False),
            ('Num workers 8 persistent pin', 8, True, True),

            ('Num workers 10', 10, False, False),
            ('Num workers 10 persistent', 10, True, False),
            ('Num workers 10 persistent pin', 10, True, True),

            ('Num workers 12', 12, False, False),
            ('Num workers 12 persistent', 12, True, False),
            ('Num workers 12 persistent pin', 12, True, True),
        ]



for name, num_workers, persistent_workers, pin_memory in tests: 
    print(name)
    benchmark_two_versions(num_workers, persistent_workers, pin_memory)
    print()