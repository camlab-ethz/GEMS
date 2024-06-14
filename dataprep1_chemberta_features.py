import pickle
import os
import torch
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import argparse
import numpy as np
from tqdm import tqdm
import time


def arg_parser():
    parser = argparse.ArgumentParser(description='Preprocess PDBbind data for DTI5')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory containing all proteins(PDB) and ligands (SDF)')
    parser.add_argument('--model', default='ChemBERTa-77M-MLM', type=str, help="Which ChemBERTa model should be used [ChemBERTa-77M-MLM, ChemBERTa-10M-MLM]")
    return parser.parse_args()

args = arg_parser()
data_dir = args.data_dir
model_descriptor = args.model

def sdf_to_smiles(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path)
    smiles_list = [Chem.MolToSmiles(mol) for mol in suppl if mol is not None]
    return smiles_list

def smiles_to_embedding(smiles, tokenizer, model):
    inputs = tokenizer(smiles, return_tensors="pt", padding=False, truncation=False)
    inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

    return embeddings.mean(dim=1).cpu()


# Device settings
#device = torch.device('cpu')
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(device)


# Load the ChemBERTa model
model_name = f'DeepChem/{model_descriptor}'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./huggingface')
model = AutoModel.from_pretrained(model_name, cache_dir='./huggingface')
model.to(device).eval()


# Initialize Log File
log_folder = os.path.join(data_dir, '.logs')
if not os.path.exists(log_folder): os.makedirs(log_folder)
log_file_path = os.path.join(log_folder, f'{model_descriptor}.txt')
log = open(log_file_path, 'a')
log.write("Generating ChemBERTa Embeddings for Proteins - Log File:\n")
log.write(f"Model Descriptor: {model_descriptor}")
log.write("\n")


# Generate a lists of all ligands
ligands = sorted([ligand for ligand in os.scandir(data_dir) if ligand.name.endswith('ligand_san.sdf')], key=lambda x: x.name)
num_ligands = len(ligands)

print(f'Number of ligands to be processed: {num_ligands}')
print(f'Model Name: {model_name}')


# Start generating embeddings for all ligands iteratively
tic = time.time()
for ligand in tqdm(ligands):

    id = ligand.name.split('_')[0]
    log_string = f'{id}: '

    save_filepath = os.path.join(data_dir, f"{id}_{model_descriptor.replace('-', '_')}.pt")
    if os.path.exists(save_filepath):
        log_string += 'Embedding already exists'
        log.write(log_string + "\n")
        continue
        
    smiles_list = sdf_to_smiles(ligand.path)
    if len(smiles_list) == 1: smiles = smiles_list[0]
    else:
        log_string += 'SMILES string could not be extracted from SDF file'
        log.write(log_string + "\n")
        continue
    
    embedding = smiles_to_embedding(smiles, tokenizer, model)

    torch.save(embedding, save_filepath)
    log_string += 'Successful'


    log.write(log_string + "\n")

print(f'Time taken for {num_ligands} ligands: {time.time() - tic} seconds')
log.close()

