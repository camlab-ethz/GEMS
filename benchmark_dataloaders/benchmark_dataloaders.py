import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import time

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
    

data_dir = '/data/grbv/PDBbind/input_graphs/training_data'

num_threads = torch.get_num_threads() // 2
torch.set_num_threads(num_threads)


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