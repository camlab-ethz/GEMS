import pickle
import os
import torch
import glob
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
log.write("Generating ChemBERTa Embeddings for Ligands - Log File:\n")
log.write(f"Model Descriptor: {model_descriptor}")
log.write("\n")


# Generate a lists of all complex IDs
complexes = sorted([compl for compl in os.scandir(data_dir) if compl.name.endswith('.pdb')], key=lambda x: x.name)

# Start generating embeddings for all ligands iteratively
tic = time.time()
num_ligands = 0
num_complexes = 0

for compl in tqdm(complexes):

    id = compl.name.split('.')[0]
    log_string = f'{id}: '

    # Find the SDF file for the current complex
    search_pattern = os.path.join(data_dir, f"{id}.sdf")
    matching_files = glob.glob(search_pattern)
    if len(matching_files) != 1:
        log_string += f'SDF file "{id}.sdf" not found (or more than one)'
        log.write(log_string + "\n")
        continue

    # Extract the smiles codes of all ligands in the SDF file
    smiles_list = sdf_to_smiles(matching_files[0])
    if len(smiles_list) < 1:
        log_string += 'No SMILES string extracted from SDF file'
        log.write(log_string + "\n")
        continue
    log.write(log_string + f"{len(smiles_list)} Ligands to Process\n")
    
    # Generate embeddings for all ligands in the SDF file
    for i, smiles in enumerate(smiles_list):
        log_string = f"--- Ligand {i+1}: "
        save_filepath = os.path.join(data_dir, f"{id}_{model_descriptor.replace('-', '_')}_L{i+1:05}.pt")
        if os.path.exists(save_filepath):
            log_string += 'Embedding already exists'
            log.write(log_string + "\n")
            continue
        
        embedding = smiles_to_embedding(smiles, tokenizer, model)
        torch.save(embedding.float(), save_filepath)
        log_string += 'Successful'
        log.write(log_string + "\n")
        num_ligands += 1
    num_complexes += 1


print(f'Time taken for {num_complexes} proteins with {num_ligands} ligands: {time.time() - tic} seconds')
log.close()

