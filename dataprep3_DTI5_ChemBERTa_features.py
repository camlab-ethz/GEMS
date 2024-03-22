import pickle
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import tqdm

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def smiles_to_embedding(smiles, tokenizer, model):
    inputs = tokenizer(smiles, return_tensors="pt", padding=False, truncation=False)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

    return embeddings.mean(dim=1)

smiles_dict = load_object('/data/grbv/PDBbind/DTI5_smiles_dict.pkl')
data_dir = '/data/grbv/PDBbind/DTI5_input_data_processed'

model_name = 'DeepChem/ChemBERTa-77M-MLM'
model_descriptor = 'ChemBERTa-77M-MLM'
embedding_size = 384
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/data/grbv/huggingface')
model = AutoModel.from_pretrained(model_name, cache_dir='/data/grbv/huggingface')

# Initialize Log File
log_file_path = os.path.join(data_dir, '.logs', f'{model_descriptor}.txt')
log = open(log_file_path, 'a')
log.write("Generating ChemBERTa Embeddings for PDBbind - Log File:\n")
log.write("\n")


num_smiles = len(smiles_dict)
print(f'Number of Smiles to be processed: {num_smiles}')
print(f'Model Name: {model_name}')
count = 0

for molecule_id, smiles in tqdm(smiles_dict.items()):

    log_string = f'{molecule_id}: '

    save_filepath = os.path.join(data_dir, molecule_id, f'{molecule_id}_{model_descriptor}.pt')
    if os.path.exists(save_filepath):
        log_string += 'Embedding already exists'
        log.write(log_string + "\n")
        continue
    

    smiles = smiles_dict[molecule_id]    
    embedding = smiles_to_embedding(smiles, tokenizer, model)

    print(molecule_id, embedding.shape)
    torch.save(embedding, save_filepath)
    log_string += 'Successful'


    log.write(log_string + "\n")

log.close()

