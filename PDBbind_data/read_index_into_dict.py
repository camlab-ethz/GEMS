import os
import json
import numpy as np

casf_2013_dir = './PDBbind/raw_data/CASF-2013/coreset'
casf_2016_dir = './PDBbind/raw_data/CASF-2016/coreset'
data_dir_general = './PDBbind/raw_data/v2020_general'
data_dir_refined = './PDBbind/raw_data/v2020_refined'

index_file = './PDBbind/index/INDEX_general_PL_data.2020'

results_general = {}
successful = 0
failed = 0
Kd_complexes = 0
Ki_complexes = 0
IC50_complexes = 0

casf_2013_complexes = [subfolder for subfolder in os.listdir(casf_2013_dir) if len(subfolder) ==4 and subfolder[0].isdigit()]
casf_2016_complexes = [subfolder for subfolder in os.listdir(casf_2016_dir) if len(subfolder) ==4 and subfolder[0].isdigit()]
general_complexes = [protein for protein in os.listdir(data_dir_general) if len(protein)==4 and protein[0].isdigit()]
refined_complexes = [protein for protein in os.listdir(data_dir_refined) if len(protein)==4 and protein[0].isdigit()]


# Open the text file for reading
with open(index_file, 'r') as file:
    # Skip the header lines
    for _ in range(6):
        next(file)

    # Read and process each line in the file
    for line in file:

        # Split the line into columns based on whitespace
        columns = line.strip().split()

        # Extract relevant information from columns
        pdb_code = columns[0]
        resolution = columns[1]
        kd_or_ki_value_str = columns[4]
        log_kd_ki = float(columns[3])
        ligand_name = columns[7].strip('()')

        if '<=' in kd_or_ki_value_str:
            type, affinity = kd_or_ki_value_str.strip().split('<=')
            precision = '<='
            value = affinity[:-2]
            unit = affinity[-2:]
        elif '>=' in kd_or_ki_value_str:
            type, affinity = kd_or_ki_value_str.strip().split('>=')
            precision = '>='
            value = affinity[:-2]
            unit = affinity[-2:]
        elif '=' in kd_or_ki_value_str:
            type, affinity = kd_or_ki_value_str.strip().split('=')
            precision = '='
            value = affinity[:-2]
            unit = affinity[-2:]
        elif '>' in kd_or_ki_value_str:
            type, affinity = kd_or_ki_value_str.strip().split('>')
            precision = '>'
            value = affinity[:-2]
            unit = affinity[-2:]
        elif '<' in kd_or_ki_value_str:
            type, affinity = kd_or_ki_value_str.strip().split('<')
            precision = '<'
            value = affinity[:-2]
            unit = affinity[-2:]
        elif '~' in kd_or_ki_value_str:
            type, affinity = kd_or_ki_value_str.strip().split('~')
            precision = '~'
            value = affinity[:-2]
            unit = affinity[-2:]
        else:
            raise Exception
            

        # Convert the numeric part to a float
        value = float(value)

        # Apply scaling factor based on the unit
        if unit == 'M':
            affinity = value
            results_general[pdb_code]={type:affinity, 'resolution':resolution, 'precision':precision, 'log_kd_ki':log_kd_ki, 'ligand_name':ligand_name}      
        elif unit == 'mM':
            affinity = value * 1e-3
            results_general[pdb_code]={type:affinity, 'resolution':resolution, 'precision':precision, 'log_kd_ki':log_kd_ki, 'ligand_name':ligand_name}
        elif unit == 'uM':
            affinity = value * 1e-6
            results_general[pdb_code]={type:affinity, 'resolution':resolution, 'precision':precision, 'log_kd_ki':log_kd_ki, 'ligand_name':ligand_name}
        elif unit == 'nM':
            affinity = value * 1e-9
            results_general[pdb_code]={type:affinity, 'resolution':resolution, 'precision':precision, 'log_kd_ki':log_kd_ki, 'ligand_name':ligand_name}
        elif unit == 'pM':
            affinity = value * 1e-12
            results_general[pdb_code]={type:affinity, 'resolution':resolution, 'precision':precision, 'log_kd_ki':log_kd_ki, 'ligand_name':ligand_name}
        elif unit == 'fM':
            affinity = value * 1e-15
            results_general[pdb_code]={type:affinity, 'resolution':resolution, 'precision':precision, 'log_kd_ki':log_kd_ki, 'ligand_name':ligand_name}
        else:
            print(pdb_code)
            raise Exception


        # Determine in which subset of the PDBbind database the complex is located
        in_casf_2013 = False
        in_casf_2016 = False
        in_general = False
        in_refined = False        
        
        dataset = []

        if pdb_code in casf_2013_complexes:
            in_casf_2013 = True
            dataset.append('casf_2013')            

        if pdb_code in casf_2016_complexes:
            in_casf_2016 = True
            dataset.append('casf_2016')

        if pdb_code in refined_complexes:
            in_refined = True
            dataset.append('refined')

        if pdb_code in general_complexes:
            in_general = True
            dataset.append('general')

        results_general[pdb_code]['dataset']=dataset


        successful+=1
        if type == 'Kd': Kd_complexes+=1
        elif type == 'Ki': Ki_complexes+=1
        elif type == 'IC50': IC50_complexes+=1
        else: print(pdb_code, type, unit)

        

print(f'Extracted Affinity Value from {successful} Complexes')
print(f'Number of Datapoints with Kd = {Kd_complexes}')
print(f'Number of Datapoints with Ki = {Ki_complexes}')
print(f'Number of Datapoints with IC50 = {IC50_complexes}')

#print(results_general)

# Save the data to a pickle file
# with open('PDBbind_data_dict2.json', 'wb') as fp:
#     json.dump(results_general, fp)

with open('PDBbind_data_dict.json', 'w', encoding='utf-8') as json_file:
    json.dump(results_general, json_file, ensure_ascii=False, indent=4)
