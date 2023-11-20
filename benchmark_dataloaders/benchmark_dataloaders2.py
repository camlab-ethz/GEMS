import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import time

class IG_Dataset(Dataset):
    def __init__(self, root, embedding=False, edge_features=False):
        super().__init__(root)

        self.data_dir = root
        self.embedding = embedding

        self.filepaths = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        #self.input_data = {ind:torch.load(filepath) for ind, filepath in enumerate(self.filepaths)}
        
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

            grph.n_nodes = grph.x.shape[0]

            self.input_data[ind] = grph
            ind += 1


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph
    


data_dir = '/cfs/earth/scratch/grbv/DTI/data/DTI_3/input_graphs_esm2_t6/training_data'



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