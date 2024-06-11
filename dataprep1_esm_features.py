from transformers import AutoModel, AutoTokenizer
import pickle
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from f_parse_pdb_general import parse_pdb
import time


def arg_parser():
    parser = argparse.ArgumentParser(description='Compute ESM embeddings for all proteins in a given directory.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory containing all proteins(PDB) and ligands (SDF)')
    parser.add_argument('--esm_checkpoint', default='t6', type=str, help="Which checkpoint of ESM should be used [t6, t12, t30, t33]")
    return parser.parse_args()
args = arg_parser()

data_dir = args.data_dir
checkpoint = args.esm_checkpoint

# Load ESM model and tokenizer from HuggingFace
if checkpoint=='t6': 
    model_name = "facebook/esm2_t6_8M_UR50D"
    model_descriptor = 'esm2_t6_8M_UR50D'
    embedding_size = 320
if checkpoint=='t12':
    model_name = "facebook/esm2_t12_35M_UR50D"
    model_descriptor = 'esm2_t12_35M_UR50D'
    embedding_size = 480
if checkpoint=='t30':
    model_name = "facebook/esm2_t30_150M_UR50D"
    model_descriptor = 'esm2_t30_150M_UR50D'
    embedding_size = 640
if checkpoint=='t33':
    model_name = "facebook/esm2_t33_650M_UR50D"
    model_descriptor = 'esm2_t33_650M_UR50D'
    embedding_size = 1280

# Initialize PDB Parser
parser = PDBParser(PERMISSIVE=1, QUIET=True)

# Device settings
#device = torch.device('cpu')
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(device)

# Load the model from HuggingFace
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Initialize Log File
log_folder = os.path.join(data_dir, '.logs')
if not os.path.exists(log_folder): os.makedirs(log_folder)
log_file_path = os.path.join(log_folder, f'{model_descriptor}.txt')
log = open(log_file_path, 'a')
log.write("Generating ESM Embeddings for Proteins - Log File:\n")
log.write(f"Model: {model_name}\n")
log.write("\n")


# Generate a lists of all proteins
proteins = sorted([protein for protein in os.scandir(data_dir) if protein.name.endswith('protein.pdb')], key=lambda x: x.name)
num_proteins = len(proteins)

print(f'Number of Proteins to be processed: {num_proteins}')
print(f'Model Name: {model_name}')


# FUNCTION TO COMPUTE EMBEDDINGS
def get_aa_embeddings_esm2(sequence, crop_EOS_BOS=True):
    token_ids = tokenizer(sequence, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        embeddings = model(token_ids).last_hidden_state
        embedding = embeddings[0]
        if crop_EOS_BOS: embedding = embedding[1:-1, :]

    return embedding    


# Start generating embeddings for all proteins iteratively 
tic = time.time()
for protein in tqdm(proteins):

    id = protein.name.split('_')[0]
    log_string = f'{id}: '

    save_filepath = os.path.join(data_dir, f'{id}_{model_descriptor}.pt')
    if os.path.exists(save_filepath):
        log_string += 'Embedding already exists'
        log.write(log_string + "\n")
        continue
    
    expected_len = 0

    # Parse the protein
    with open(protein.path) as pdbfile:
        prot = parse_pdb(parser, id, pdbfile)    


    emb = np.array([], dtype=np.int64).reshape(0,embedding_size)

    for chain in prot:

        chain_comp = prot[chain]['composition']
        if chain_comp == [True, False] or chain_comp == [True, True]:
            
            sequence = prot[chain]['aa_seq']
            expected_len += len(sequence)

            try:
                embeddings = get_aa_embeddings_esm2(sequence)
                emb = np.vstack((emb, embeddings.cpu()))
                
            except Exception as e:
                log_string += str(e)
                break



    if not emb.shape == (expected_len, embedding_size):
        log_string += f'Embedding has wrong shape {emb.shape} instead of ({expected_len}x{embedding_size})'
        log.write(log_string + "\n")
        continue
    else:
        torch.save(emb, save_filepath)
        log_string += 'Successful'

    log.write(log_string + "\n")

print(f'Time taken for {num_proteins} proteins: {time.time() - tic} seconds')
log.close()