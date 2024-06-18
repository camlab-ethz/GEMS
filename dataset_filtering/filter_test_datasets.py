import json
import os

# The path to the JSON file
json_file_path = '/home/grabeda2/DTI/dataset_filtering/C4/c4_test_dataset_number_of_similarities.json'

casf2013_dir = f'/data/grbv/PDBbind/DTI_5/input_graphs_ankh_base_unpad/test_data/casf2013_c4'
casf2016_dir = f'/data/grbv/PDBbind/DTI_5/input_graphs_ankh_base_unpad/test_data/casf2016_c4'

# Read the JSON file and convert it into a dictionary
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    similarities = json.load(json_file)

for complex in similarities.keys():
    
    try:
        os.remove(f'{casf2013_dir}/{complex}_graph_ankh_base.pt')
        print(f'Deleted {complex}')
    except FileNotFoundError:
        pass
    
    try:
        os.remove(f'{casf2016_dir}/{complex}_graph_ankh_base.pt')
        print(f'Deleted {complex}')
    except FileNotFoundError:
        pass

