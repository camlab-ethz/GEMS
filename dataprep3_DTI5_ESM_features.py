from transformers import AutoModel, AutoTokenizer
import pickle
import os
import torch
import numpy as np

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Load ESM model and tokenizer from HuggingFace
#model_name = "facebook/esm2_t6_8M_UR50D"
#model_name = "facebook/esm2_t12_35M_UR50D"
#model_name = "facebook/esm2_t30_150M_UR50D"
model_name = "facebook/esm2_t33_650M_UR50D"

model_descriptor = 'esm2_t33_650M'
embedding_size = 1280

device_idx = 2
torch.cuda.set_device(device_idx)
device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
print(device, torch.cuda.current_device(), torch.cuda.get_device_name())

model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_dir = '/data/grbv/PDBbind/DTI5_input_data_processed'

# Initialize Log File
log_file_path = os.path.join(data_dir, '.logs', f'{model_descriptor}.txt')
log = open(log_file_path, 'a')
log.write("Generating ESM Embeddings for PDBbind - Log File:\n")
log.write("Data: PDBbind v2020 refined and general set merged\n")
log.write("\n")

# Generate a lists of all protein-ligand complexes, the corresponding folder path and protein_dictionary paths
proteins = [subfolder for subfolder in os.listdir(data_dir) if len(subfolder) ==4 and subfolder[0].isdigit()]
folder_paths = [os.path.join(data_dir, protein) for protein in proteins]
protein_paths = [os.path.join(folder_path, f'{protein}_protein_dict.pkl') for protein, folder_path in zip(proteins, folder_paths)]


num_proteins = len(proteins)
print(f'Number of Proteins to be processed: {num_proteins}')
print(f'Model Name: {model_name}')
count = 0


# FUNCTION TO COMPUTE EMBEDDINGS
def get_aa_embeddings_esm2(sequence, crop_EOS_BOS=True):
    token_ids = tokenizer(sequence, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        embeddings = model(token_ids).last_hidden_state
        embedding = embeddings[0]
        if crop_EOS_BOS: embedding = embedding[1:-1, :]

    return embedding    

# ind = proteins.index('4ycu')
# for id, folder_path, protein_path in zip(proteins[ind:ind+1], folder_paths[ind:ind+1], protein_paths[ind:ind+1]):
for id, folder_path, protein_path in zip(proteins, folder_paths, protein_paths):

    log_string = f'{id}: '
    #count+=1
    #print(f'{count}/{num_proteins}')

    save_filepath = os.path.join(folder_path, f'{id}_{model_descriptor}.pt')
    if os.path.exists(save_filepath):
        log_string += 'Embedding already exists'
        log.write(log_string + "\n")
        continue

    print(id)
    
    expected_len = 0

    protein = load_object(protein_path)
    emb = np.array([], dtype=np.int64).reshape(0,embedding_size)

    for chain in protein:

        chain_comp = protein[chain]['composition']
        if chain_comp == [True, False] or chain_comp == [True, True]:
            
            sequence = protein[chain]['aa_seq']
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

log.close()