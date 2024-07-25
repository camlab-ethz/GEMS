import os
import json

# Load dict from json
with open('./PDBbind_data_splits/PDBbind_c0_data_split.json', 'r') as json_file:
    split_dict_c0 = json.load(json_file)

with open('./PDBbind_data_splits/PDBbind_c4_data_split.json', 'r') as json_file:
    split_dict_c4 = json.load(json_file)


casf2013_c4 = [file[0:4] for file in os.listdir('/cluster/work/math/dagraber/DTI/PDBbind0/test_data/ankh_base/casf2013_c4') if file[0].isdigit() and file.endswith('.pt')]
casf2016_c4 = [file[0:4] for file in os.listdir('/cluster/work/math/dagraber/DTI/PDBbind0/test_data/ankh_base/casf2016_c4') if file[0].isdigit() and file.endswith('.pt')]
print(len(casf2013_c4))
print(len(casf2016_c4))

split_dict_c4['casf2016_c4'] = casf2016_c4
split_dict_c4['casf2013_c4'] = casf2013_c4

split_dict_c0['casf2016_c4'] = casf2016_c4
split_dict_c0['casf2013_c4'] = casf2013_c4


with open('./PDBbind_data_splits/PDBbind_c0_data_split2.json', 'w', encoding='utf-8') as json_file:
    json.dump(split_dict_c0, json_file, ensure_ascii=False, indent=4)

with open('./PDBbind_data_splits/PDBbind_c4_data_split2.json', 'w', encoding='utf-8') as json_file:
    json.dump(split_dict_c4, json_file, ensure_ascii=False, indent=4)