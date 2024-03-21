import pickle
import os
import ankh
import torch
import numpy as np


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


model_name = 'ankh_large'
model_descriptor = 'ankh_large'
embedding_size = 1536

device_idx = 4
torch.cuda.set_device(device_idx)
device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
print(device, torch.cuda.current_device(), torch.cuda.get_device_name())

if model_name == 'ankh_base': model, tokenizer = ankh.load_base_model()
elif model_name == 'ankh_large': model, tokenizer = ankh.load_large_model()
model.to(device).eval()

data_dir = '/data/grbv/PDBbind/DTI5_input_data_processed'

# Initialize Log File
log_file_path = os.path.join(data_dir, '.logs', f'{model_descriptor}.txt')
log = open(log_file_path, 'a')
log.write("Generating ANKH Embeddings for PDBbind - Log File:\n")
log.write("\n")

# Generate a lists of all protein-ligand complexes, the corresponding folder path and protein_dictionary paths
proteins = sorted([subfolder for subfolder in os.listdir(data_dir) if len(subfolder) ==4 and subfolder[0].isdigit()])
folder_paths = [os.path.join(data_dir, protein) for protein in proteins]
protein_paths = [os.path.join(folder_path, f'{protein}_protein_dict.pkl') for protein, folder_path in zip(proteins, folder_paths)]


num_proteins = len(proteins)
print(f'Number of Proteins to be processed: {num_proteins}')
print(f'Model Name: {model_name}')
count = 0


# FUNCTION TO COMPUTE EMBEDDINGS

def get_aa_embeddings_ankh(protein_sequence):

    protein_sequences = [list(protein_sequence)]
    outputs = tokenizer.batch_encode_plus(protein_sequences, 
                                add_special_tokens=False, 
                                padding=True, 
                                is_split_into_words=True, 
                                return_tensors="pt")
    outputs.to(device)

    with torch.no_grad():
        embedding = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask']).last_hidden_state

    return embedding.squeeze() 


# ind = proteins.index('1a0q')
# for id, folder_path, protein_path in zip(proteins[ind:ind+1], folder_paths[ind:ind+1], protein_paths[ind:ind+1]):
for id, folder_path, protein_path in zip(proteins, folder_paths, protein_paths):

    log_string = f'{id}: '

    save_filepath = os.path.join(folder_path, f'{id}_{model_descriptor}.pt')
    if os.path.exists(save_filepath):
        log_string += 'Embedding already exists'
        log.write(log_string + "\n")
        continue

    
    expected_len = 0

    protein = load_object(protein_path)
    emb = np.array([], dtype=np.int64).reshape(0,embedding_size)

    for chain in protein:

        chain_comp = protein[chain]['composition']
        if chain_comp == [True, False] or chain_comp == [True, True]:
            
            sequence = protein[chain]['aa_seq']
            expected_len += len(sequence)

            try:
                embeddings = get_aa_embeddings_ankh(sequence)
                emb = np.vstack((emb, embeddings.cpu()))
                
            except Exception as e:
                log_string += str(e)
                break



    if not emb.shape == (expected_len, embedding_size):
        log_string += f'Embedding has wrong shape {emb.shape} instead of ({expected_len}x{embedding_size})'
        log.write(log_string + "\n")
        continue
    else:
        #esm = torch.load(save_filepath.replace('ankh_base', 'esm2_t6_8M'))
        print(id, emb.shape)#, esm.shape)
        torch.save(emb, save_filepath)
        log_string += 'Successful'

    log.write(log_string + "\n")

log.close()