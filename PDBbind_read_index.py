import os
import pickle
import numpy as np

index_file = '/data/grbv/PDBbind/index/INDEX_general_PL_data.2020'
output_data_dir = '/data/grbv/PDBbind/'

results_general = {}
successful = 0
failed = 0
Kd_complexes = 0
Ki_complexes = 0
IC50_complexes = 0

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


        successful+=1
        if type == 'Kd': Kd_complexes+=1
        elif type == 'Ki': Ki_complexes+=1
        elif type == 'IC50': IC50_complexes+=1
        else: print(pdb_code, type, unit)

        

print(f'Extracted Affinity Value from {successful} Complexes')
print(f'Number of Datapoints with Kd = {Kd_complexes}')
print(f'Number of Datapoints with Ki = {Ki_complexes}')
print(f'Number of Datapoints with IC50 = {IC50_complexes}')

print(results_general)

# Save the data to a pickle file
with open(os.path.join(output_data_dir, 'DTI5_general_affinity_dict.pkl'), 'wb') as fp:
    pickle.dump(results_general, fp)