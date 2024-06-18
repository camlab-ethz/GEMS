import pickle
import os
import ankh
import torch
import numpy as np
import argparse
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from f_parse_pdb_general import parse_pdb
import time


def arg_parser():
    parser = argparse.ArgumentParser(description='Preprocess PDBbind data for DTI5')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory containing all proteins(PDB) and ligands (SDF)')
    parser.add_argument('--ankh_base', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], help="If the ankh_base model should be used")
    return parser.parse_args()

args = arg_parser()

data_dir = args.data_dir
model_name = 'ankh_base' if args.ankh_base else 'ankh_large'

# Initialize PDB Parser
parser = PDBParser(PERMISSIVE=1, QUIET=True)


# Device settings
#device = torch.device('cpu')
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(device)

# Load the model from ANKH
if model_name == 'ankh_base':
    model, tokenizer = ankh.load_base_model()
    embedding_size = 768
elif model_name == 'ankh_large':
    model, tokenizer = ankh.load_large_model()
    embedding_size = 1536

model.to(device).eval()



# Initialize Log File
log_folder = os.path.join(data_dir, '.logs')
if not os.path.exists(log_folder): os.makedirs(log_folder)
log_file_path = os.path.join(log_folder, f'{model_name}.txt')
log = open(log_file_path, 'a')
log.write("Generating ANKH Embeddings - Log File:\n")
log.write(f"Model Name: {model_name}")
log.write("\n")


# Generate a lists of all proteins
proteins = sorted([protein for protein in os.scandir(data_dir) if protein.name.endswith('protein.pdb')], key=lambda x: x.name)
num_proteins = len(proteins)

print(f'Number of Proteins to be processed: {num_proteins}')
print(f'Model Name: {model_name}')


# FUNCTION TO COMPUTE EMBEDDINGS
def get_aa_embeddings_ankh(protein_sequence):

    protein_sequences = [list(protein_sequence)]
    inputs = tokenizer.batch_encode_plus(protein_sequences, 
                                add_special_tokens=False, 
                                padding=True, 
                                is_split_into_words=True, 
                                return_tensors="pt")
    inputs.to(device)

    with torch.no_grad():
        embedding = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state

    return embedding.squeeze() 


# Start generating embeddings for all proteins iteratively 
tic = time.time()
for protein in tqdm(proteins):

    id = protein.name.split('_')[0]
    log_string = f'{id}: '

    save_filepath = os.path.join(data_dir, f'{id}_{model_name}.pt')
    if os.path.exists(save_filepath):
        log_string += 'Embedding already exists'
        log.write(log_string + "\n")
        continue

    expected_len = 0

    # Parse the protein
    with open(protein.path) as pdbfile:
        prot = parse_pdb(parser, id, pdbfile)    
        
    #emb = np.array([], dtype=np.int64).reshape(0,embedding_size)
    emb = torch.empty(0, embedding_size, dtype=torch.float)

    for chain in prot:

        chain_comp = prot[chain]['composition']
        if chain_comp == [True, False] or chain_comp == [True, True]:
            
            sequence = prot[chain]['aa_seq']
            expected_len += len(sequence)

            try:
                embeddings = get_aa_embeddings_ankh(sequence)
                emb = torch.vstack((emb, embeddings.cpu()))
                
            except Exception as e:
                log_string += str(e)
                break



    if not emb.shape == (expected_len, embedding_size):
        log_string += f'Embedding has wrong shape {emb.shape} instead of ({expected_len}x{embedding_size})'
        log.write(log_string + "\n")
        continue
    else:
        torch.save(emb.float(), save_filepath)
        log_string += 'Successful'

    log.write(log_string + "\n")

print(f'Time taken for {num_proteins} proteins: {time.time() - tic} seconds')
log.close()

