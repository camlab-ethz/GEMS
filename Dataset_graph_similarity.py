import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class IG_Dataset(Dataset):
    def __init__(self, root, embedding=False, 
                             edge_features=False, 
                             atom_features=False, 
                             refined_only=False,
                             exclude_ic50=False,
                             exclude_nmr=False,
                             resolution_threshold=5.,
                             precision_strict=False):
        
        super().__init__(root)

        self.data_dir = root
        self.filepaths = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]
        self.input_data = {}
        
        # Process all the graphs according to kwargs
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



            # Delete the masternode from the graph
            x = x[:-1]
            
               
            train_graph = Data(x = x, 
                            edge_index=grph.edge_index, 
                            edge_attr=grph.edge_attr, 
                            y=pK
                            ,pos=grph.pos
                            ,id=grph.id
                            )


            self.input_data[ind] = train_graph
            ind += 1


    def len(self):
        return len(self.input_data)
    
    def get(self, idx):
        graph = self.input_data[idx]
        return graph