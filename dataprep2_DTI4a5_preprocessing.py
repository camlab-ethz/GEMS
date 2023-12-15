from Bio.PDB.PDBParser import PDBParser
from f_parse_pdb_general import parse_pdb
import numpy as np
import cupy as cp
from rdkit import Chem
import os
from f_connect_to_accessible_aa import connect_to_accessible_aa
import shutil
import pickle


def parse_sdf_file(file_path):
    suppl = Chem.SDMolSupplier(file_path, sanitize = True, removeHs=True, strictParsing=True)
    molecules = []
    for mol in suppl:
        if mol is not None:
            molecules.append(mol)
    return molecules

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

input_data_dir_general = '/data/grbv/PDBbind/raw_data/v2020_general_san'
input_data_dir_refined = '/data/grbv/PDBbind/raw_data/v2020_refined_san'
output_data_dir = '/data/grbv/PDBbind/DTI_input_data_processed/'
affinity_dict = load_object('/data/grbv/PDBbind/v2020_general_affinity_dict.pkl')

if not os.path.exists(output_data_dir): os.makedirs(output_data_dir)

protein_ids_general = [protein for protein in os.listdir(input_data_dir_general) if len(protein)==4 and protein[0].isdigit()]
protein_ids_refined = [protein for protein in os.listdir(input_data_dir_refined) if len(protein)==4 and protein[0].isdigit()]
protein_ids = protein_ids_general + protein_ids_refined

protein_folders_general = [os.path.join(input_data_dir_general, protein_id) for protein_id in protein_ids_general]
protein_folders_refined = [os.path.join(input_data_dir_refined, protein_id) for protein_id in protein_ids_refined]

ligand_paths_general = [os.path.join(protein_folder, ligand_file) for protein_folder in protein_folders_general 
                for ligand_file in os.listdir(protein_folder) if ligand_file.endswith('san.sdf')]
ligand_paths_refined = [os.path.join(protein_folder, ligand_file) for protein_folder in protein_folders_refined 
                for ligand_file in os.listdir(protein_folder) if ligand_file.endswith('san.sdf')]
ligand_paths = ligand_paths_general + ligand_paths_refined


protein_paths_general = [os.path.join(protein_folder, protein_file) for protein_folder in protein_folders_general 
                for protein_file in os.listdir(protein_folder) if protein_file.endswith('protein.pdb')]
protein_paths_refined = [os.path.join(protein_folder, protein_file) for protein_folder in protein_folders_refined 
                for protein_file in os.listdir(protein_folder) if protein_file.endswith('protein.pdb')]
protein_paths = protein_paths_general + protein_paths_refined


count = 0
len_prot = len(protein_paths)
len_lig = len(ligand_paths)

print(f'Number of Protein PDBs: {len_prot}')
print(f'Number of Ligand SDFs: {len_lig}')


# Initialize Log File:
log_folder = output_data_dir + '.logs/'
if not os.path.exists(log_folder): os.makedirs(log_folder)
log_file_path = os.path.join(log_folder, "preprocessing_DTI4a5.txt")
log = open(log_file_path, 'a')
log.write("Data Preprocessing PDBbind - Log File:\n")
log.write("Data: PDBbind v2020 refined and general set merged\n")
log.write("\n")


# Initialize PDB Parser
parser = PDBParser(PERMISSIVE=1, QUIET=True)

amino_acids = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
known_hetatms = ['ZN','MG','NA','MN','CA','K','NI','FE','CO','HG','CD','CU','CS','AU','LI','GA','IN','BA','RB','SR']
known_residues = amino_acids + known_hetatms


# Here start a loop over the complexes
#----------------------------------------------------------
# ind = protein_ids.index('4ycu')
# for protein_id, ligand_path, protein_path in zip(protein_ids[ind:ind+1], ligand_paths[ind:ind+1], protein_paths[ind:ind+1]):
for protein_id, ligand_path, protein_path in zip(protein_ids, ligand_paths, protein_paths):
    
    log_string = f'{protein_id}: '
    count+=1
    print(f'{count}/{len_prot}')
    
    # PRESELECTION OF COMPLEXES
    # -----------------------------------------------------

    # TEST 1: AFFINITY DATA - Continue only if there is a valid affinity value available for this complex
    if protein_id not in affinity_dict.keys():
        log_string += 'Protein is not processed: No valid affinity value'
        log.write(log_string + "\n")
        continue

    # TEST 2: LIGAND PARSING - Continue only if the parsed ligand has been processed successfully, else skip this complex
    ligand = parse_sdf_file(ligand_path)
    if len(ligand) == 1: mol = ligand[0]
    else:
        log_string += 'Ligand could not be parsed successfully'
        log.write(log_string + "\n")
        continue
    
    # TEST 3: LIGAND SIZE - Get coordinate Matrix of the ligand molecule - Continue only if the ligand has more than 6 heavy atoms
    conformer = mol.GetConformer()
    coordinates = conformer.GetPositions()
    pos = np.array(coordinates)
    if pos.shape[0]<5:
        log_string += 'Ligand is smaller than 5 Atoms and is therefore skipped'
        log.write(log_string + "\n")
        continue
    


    # PARSING OF PROTEIN PDB TO GENERATE COORDINATE MATRIX
    # -----------------------------------------------------

    with open(protein_path) as pdbfile:
        protein = parse_pdb(parser, protein_id, pdbfile)


    protein_atomcoords = np.array([], dtype=np.int64).reshape(0,3)
    res_list = []
    residue_memberships = []
        
    clean_aa_chain = False
    chain_too_long = False
    residue_idx = 1

    # Iterate over the chains in the protein
    for chain in protein:

        chain_comp = protein[chain]['composition']

        # CHAIN CONTAINS ONLY AMINO ACIDS (AND HETATMS in chain)
        if chain_comp == [True, False] or chain_comp == [True, True]:
            clean_aa_chain = True

            # If the chain is longer than 1024, skip the complex
            if len(protein[chain]['aa_seq']) > 1022:
                chain_too_long = True
                break
            
            for residue in protein[chain]['aa_residues']:
                res_dict = protein[chain]['aa_residues'][residue]

                # Append the coords of the residue to the protein_atomcoords
                protein_atomcoords = np.vstack((protein_atomcoords, res_dict['coords']))

                res_list.append((residue_idx, res_dict['resname']))
                
                memb = [residue_idx for atom in res_dict['atom_indeces']]
                residue_memberships.extend(memb)

                residue_idx += 1
        
        # CHAIN CONTAINS HETATMS BUT NO AMINO ACIDS
        elif protein[chain]['composition'] == [False, True]: 
            
            for hetatm_res in protein[chain]['hetatm_residues']:
                hetatmres_dict = protein[chain]['hetatm_residues'][hetatm_res]

                # Append the coords of the residue to the protein_atomcoords
                protein_atomcoords = np.vstack((protein_atomcoords, hetatmres_dict['hetatmcoords']))

                res_list.append((residue_idx, hetatmres_dict['resname']))

                memb = [residue_idx for atom in hetatmres_dict['atoms']]
                residue_memberships.extend(memb)

                residue_idx +=1
        
        # CHAIN CONTAINS ONLY WATER MOLECULES        
        elif protein[chain]['composition'] == [False, False]:
            pass



    # If a chain is too long to generate an ESM Embedding, skip the complex
    if chain_too_long:
        log_string += 'Protein AA sequence too long for ESM'
        log.write(log_string + "\n")
        continue


    

    # COMPUTE CONNECTIVITY BETWEEN LIGAND AND PROTEIN ATOMS
    # -----------------------------------------------------

    if not clean_aa_chain:
        log_string += 'No clean AA_chain has been found, complex is skipped'
        log.write(log_string + "\n")
        continue


    # Prepare data for creating connections with f_connect_to_accessible_aa.py
    residue_membership = np.array(residue_memberships)

    # Make a preselection of the enzyme atoms, only include those that are closer than 6A from any of the substrate atoms
    max_len = 5
    diff = protein_atomcoords[np.newaxis, :, :] - pos[:, np.newaxis, :]
    pairwise_distances = np.linalg.norm(diff, axis=2)
    close = pairwise_distances <= max_len + 1 # THIS NEEDS TO BE +1 to include as many atoms in the preselection as in DTI4b
    chosen = np.any(close, axis = 0)
    

    # Create selection of atomscoords based on 'chosen' and add the coords and the residue membership (=0) of the substrate atoms
    # to the residue membership, as the substrate atoms should also be included in the neighborhood search (vectors will be blocked by neighboring atoms)
    sel = protein_atomcoords[chosen]
    residue_membership_sel = residue_membership[chosen]

    # Add residue membership = 0 to the residue membership array for each atom of the ligand
    residue_membership_combined = np.concatenate([residue_membership_sel, np.zeros(pos.shape[0], dtype=int)])
    

    # RUN STARLINE ALGORITHM ON GPU
    #------------------------------------------------------------------
    selected_atoms = connect_to_accessible_aa(pos, sel, max_len=max_len, device_idx=1)
    #------------------------------------------------------------------


    # From the connectivity matrix (n atoms ligand x n atoms protein sel), derive the connections
    connections = [residue_membership_combined[star] for star in selected_atoms]
    connections = [list(set(connection)) for connection in connections]
    for l in connections: l.remove(0)

    # Check if any of the connected residues are unknown. If yes, skip that protein
    connected_res_num = sorted(list(set([atm for l in connections for atm in l])))
    connected_res_name = [res_list[aa-1][1] for aa in connected_res_num]


    unknown_res = [(res not in known_residues and res.strip('0123456789') not in known_residues) for res in connected_res_name]
    if any(unknown_res):
        log_string += 'Ligand has been connected to a unknown protein residue, the complex is therefore skipped'
        log.write(log_string + "\n")
        continue


    
    # EXPORT DATA
    # -------------------------------------------------------
    save_dir = os.path.join(output_data_dir, protein_id)
    if not os.path.exists(save_dir): os.makedirs(save_dir)


    # Move corresponding SDF file to output folder
    #shutil.copy(ligand_path, save_dir)


    # Export protein dictionary as pkl
    #filepath = os.path.join(save_dir, f'{protein_id}_protein_dict.pkl')

    #with open(filepath, 'wb') as fp:
    #    pickle.dump(protein, fp)
        #print('Protein Dictionary saved successfully to file')
    

    # Export CONNECTIONS as dict
    connections_dict = {'connections':connections, 'res_num':connected_res_num, 'res_name':connected_res_name}
    filepath = os.path.join(save_dir, f'{protein_id}_connections_DTI4a5.pkl')

    with open(filepath, 'wb') as fp:
        pickle.dump(connections_dict, fp)
        #print('Connections Dictionary saved successfully to file')

    log_string += 'Successful'
    log.write(log_string + "\n")

log.close()