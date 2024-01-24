import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import umap


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Load all necessary data
smiles_dict = load_object('/data/grbv/PDBbind/DTI5_smiles_dict.pkl')
clustering_output = 'clusterRes_cluster_DTI5_1.tsv'
folder_path = '/data/grbv/PDBbind/DTI_5_1'

casf2016_dir = '/data/grbv/PDBbind/DTI_5/input_graphs_esm2_t6_8M/test_data/casf2016'
casf2013_dir = '/data/grbv/PDBbind/DTI_5/input_graphs_esm2_t6_8M/test_data/casf2013'

casf2016_complexes = [filename[0:4] for filename in os.listdir(casf2016_dir) if 'graph' in filename]
casf2013_complexes = [filename[0:4] for filename in os.listdir(casf2013_dir) if 'graph' in filename]

test_complexes = casf2013_complexes + casf2016_complexes



morgan_fingerprints_dict = {compl:{} for compl in smiles_dict}

for compl in smiles_dict:
    morgan_fingerprints_dict[compl]['smiles'] = smiles_dict[compl] 
    morgan_fingerprints_dict[compl]['testset'] = compl in test_complexes 



# Function to convert SMILES to Morgan fingerprint
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
    else:
        return None

# Generate fingerprints
    
for compl in morgan_fingerprints_dict:
    fp = smiles_to_fingerprint(morgan_fingerprints_dict[compl]['smiles'])
    if fp is not None:
        morgan_fingerprints_dict[compl]['morgan'] = list(fp)


# Save the data to a pickle file
with open('DTI5_morgan_fp_dict.pkl', 'wb') as fp:
    pickle.dump(morgan_fingerprints_dict, fp)