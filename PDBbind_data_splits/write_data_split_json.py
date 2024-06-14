import os
import json


casf2016 = [folder for folder in os.listdir('./PDBbind/raw_data/CASF-2016/coreset') if len(folder) == 4 and folder[0].isdigit()]
casf2013 = [folder for folder in os.listdir('./PDBbind/raw_data/CASF-2013/coreset') if len(folder) == 4 and folder[0].isdigit()]


split_dict_c4 = {'casf2016': casf2016, 'casf2013': casf2013}
split_dict_c0 = {'casf2016': casf2016, 'casf2013': casf2013}

training_set_c4 = [file[0:4] for file in os.listdir('./training_data/DTI5c4/training_data') if file[0].isdigit() and file.endswith('.pt')]
print(len(training_set_c4))
split_dict_c4['train'] = training_set_c4

training_set_c0_1 = [folder for folder in os.listdir('./PDBbind/raw_data/v2020_general') if len(folder) == 4 and folder[0].isdigit()]
training_set_c0_2 = [folder for folder in os.listdir('./PDBbind/raw_data/v2020_refined') if len(folder) == 4 and folder[0].isdigit()]
training_set_c0 = training_set_c0_1 + training_set_c0_2
print(len(training_set_c0))
split_dict_c0['train'] = training_set_c0


with open('./PDBbind_data_splits/PDBbind_c0_data_split.json', 'w', encoding='utf-8') as json_file:
    json.dump(split_dict_c0, json_file, ensure_ascii=False, indent=4)

with open('./PDBbind_data_splits/PDBbind_c4_data_split.json', 'w', encoding='utf-8') as json_file:
    json.dump(split_dict_c4, json_file, ensure_ascii=False, indent=4)