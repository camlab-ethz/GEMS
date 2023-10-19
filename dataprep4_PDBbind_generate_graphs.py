import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges
import os
import torch
from torch_geometric.utils import to_undirected, add_self_loops
from rdkit.Chem.MolStandardize import rdMolStandardize  
from torch_geometric.data import Data, Batch
from f_helper_functions import load_object


def parse_sdf_file(file_path):
    suppl = Chem.SDMolSupplier(file_path)#, sanitize = False)
    molecules = []
    for mol in suppl:
        if mol is not None:
            molecules.append(mol)
    return molecules

salt_remover = rdMolStandardize.FragmentRemover() 


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element.
    Unlike `one_of_k_encoding`, if `x` is not in `allowable_set`, this method
    pretends that `x` is the last element of `allowable_set`.
    Parameters
    ----------
    x: object
        Must be present in `allowable_set`.
    allowable_set: list
        List of allowable quantities.
    Examples
    --------
    >>> dc.feat.graph_features.one_of_k_encoding_unk("s", ["a", "b", "c"])
    [False, False, True]
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def one_of_k_encoding(x, allowable_set):
    """Encodes elements of a provided set as integers.
    Parameters
    ----------
    x: object
        Must be present in `allowable_set`.
    allowable_set: list
        List of allowable quantities.
    Example
    -------
    >>> import deepchem as dc
    >>> dc.feat.graph_features.one_of_k_encoding("a", ["a", "b", "c"])
    [True, False, False]
    Raises
    ------
    `ValueError` if `x` is not in `allowable_set`.
    """
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


all_atoms = ['H','Li','B','C','N','O','F','Na','Mg','Si','P','S','Cl','K','Ca','V','Mn','Fe','Co','Ni','Cu','Zn','Ga','As','Se','Br','Rb','Sr','Ru','Rh','Cd','In','Sb','Te','I','Cs','Ba','Os','Ir','Pt','Au','Hg'] #42

amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
               "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

hetatm_smiles_dict1 = {'ZN': '[Zn+2]', 'MG': '[Mg+2]', 'NA': '[Na+1]', 'MN': '[Mn+2]', 'CA': '[Ca+2]', 'K': '[K+1]',
                      'NI': '[Ni+2]', 'FE': '[Fe+2]', 'CO': '[Co+2]', 'HG': '[Hg+2]', 'CD': '[Cd+2]', 'CU': '[Cu+2]', 
                      'CS': '[Cs+1]', 'AU': '[Au+1]', 'LI': '[Li+1]', 'GA': '[Ga+3]', 'IN': '[In+3]', 'BA': '[Ba+2]',
                      'RB': '[Rb+1]', 'SR': '[Sr+2]'}

hetatm_smiles_dict2 = {'Zn': '[Zn+2]', 'Mg': '[Mg+2]', 'Na': '[Na+1]', 'Mn': '[Mn+2]', 'Ca': '[Ca+2]', 'K': '[K+1]',
                      'Ni': '[Ni+2]', 'Fe': '[Fe+2]', 'Co': '[Co+2]', 'Hg': '[Hg+2]', 'Cd': '[Cd+2]', 'Cu': '[Cu+2]', 
                      'Cs': '[Cs+1]', 'Au': '[Au+1]', 'Li': '[Li+1]', 'Ga': '[Ga+3]', 'In': '[In+3]', 'Ba': '[Ba+2]',
                      'Rb': '[Rb+1]', 'Sr': '[Sr+2]'}



def make_undirected_with_self_loops(edge_index, edge_attr, undirected = True, self_loops = True):

    self_loop_feature_vector = torch.tensor(   [0., 1., 0.,                         # it's a self-loop
                                                0,                                  # length is zero
                                                0., 0., 0.,0.,0.,                   # bondtype = None
                                                0.,                                 # is not conjugated
                                                0.,                                 # is not in ring
                                                0., 0., 0., 0., 0., 0.])            # No stereo -> self-loop 

    if undirected: edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    if self_loops: edge_index, edge_attr = add_self_loops( edge_index, edge_attr, fill_value = self_loop_feature_vector)

    return edge_index, edge_attr



def atom_features(mol, padding_len): # padding: first append n zeros (length of amino acid embeddings)

    x = []
    rdPartialCharges.ComputeGasteigerCharges(mol)

    for atom in mol.GetAtoms():

        padding = [0 for n in range(padding_len)]
        symbol = atom.GetSymbol()
        ringm = [atom.IsInRing()]
        hybr = atom.GetHybridization()
        charge = [float(atom.GetFormalCharge())] 
        #charge = [float(atom.GetProp('_GasteigerCharge'))] 
        aromatic = [atom.GetIsAromatic()]
        mass = [atom.GetMass()/100]
        numHs = atom.GetTotalNumHs()
        degree = atom.GetDegree()
        chirality = str(atom.GetChiralTag())

        #print(symbol, ringm, hybr, charge, aromatic, mass, numHs, degree, chirality)

        results =   padding + \
                    one_of_k_encoding_unk(symbol, all_atoms) + \
                    ringm  + \
                    one_of_k_encoding_unk(hybr, [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) + \
                    charge + \
                    aromatic + \
                    mass + \
                    one_of_k_encoding(numHs, [0, 1, 2, 3, 4]) + \
                    one_of_k_encoding(degree,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                    one_of_k_encoding_unk(chirality, ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'OTHER'])     
        
        x.append(results)  

    return np.array(x)



def edge_index_and_attr(mol, pos, undirected = True, self_loops = True):

    edge_index = [[],[]]
    edge_attr = []

    #  Edge Attributes - Loop over edges and compute feature vector
    #--------------------------------------------------------------------    
    for bond in mol.GetBonds():

        atm1 = bond.GetBeginAtomIdx()
        atm2 = bond.GetEndAtomIdx()

        edge_index[0].append(atm1)
        edge_index[1].append(atm2)


        # Generate Edge Feature Vector
        #--------------------------------------------------------------------
        edge_feature_vector = []
        #print(f'Bond {bond.GetIdx()} between atoms {atm1, atm2}')

        # Edge Type (covalent bond, non-covalent_bond, self-loop)
        #print('---Covalent/Self-Loop/Non-Covalent: ', one_of_k_encoding_unk('covalent', ['covalent','self-loop','non-covalent']))
        edge_feature_vector.extend(one_of_k_encoding('covalent', ['covalent','self-loop','non-covalent']))

        # Length of Edge
        length = np.linalg.norm(pos[atm1]-pos[atm2])
        #print('---Bond Length: ', length )
        edge_feature_vector.append(length/10)

        # Bond Type (single, double, aromatic)
        #print('---Bond Type: ', one_of_k_encoding_unk(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 'non-covalent']))
        edge_feature_vector.extend(one_of_k_encoding(bond.GetBondTypeAsDouble(), [0.,1.0,1.5,2.0,3.0]))

        # Conjugated
        #print('---Is Conjugated: ', [bond.GetIsConjugated()])
        edge_feature_vector.append(bond.GetIsConjugated())

        # Is in Ring?
        #print('---Is in Ring: ', [bond.IsInRing()])
        edge_feature_vector.append(bond.IsInRing())

        # Stereo
        allowed = [Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOANY, 
                Chem.rdchem.BondStereo.STEREOE, 
                Chem.rdchem.BondStereo.STEREOZ, 
                Chem.rdchem.BondStereo.STEREOCIS, 
                Chem.rdchem.BondStereo.STEREOTRANS]
        
        #print('---Bond Stereo: ', one_of_k_encoding(bond.GetStereo(), allowed))
        edge_feature_vector.extend(one_of_k_encoding(bond.GetStereo(), allowed))

        edge_attr.append(edge_feature_vector)

    # Make undirected and add self loops if necessary
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float64)
    edge_index, edge_attr = make_undirected_with_self_loops(edge_index, edge_attr, undirected=undirected, self_loops=self_loops)
    return edge_index, edge_attr



#-------------------------------------------------------------------------------------------------------------
# GENERATE INTERACTION-GRAPHS OF ALL COMPLEXES IN PDBbind FOLDER
# - dataprep2 needs to have run to generate protein_dicts and connections_dicts
# - dataprep3 needs to have run to generate ESM embeddings
#------------------------------------------------------------------------------------------------------------- 

# Choose the esm embedding that should be used:
embedding_descriptor = 'esm2_t6_8M'
num_atomfeatures = 71
output_folder = '/data/grbv/PDBbind/input_graphs/'


# GET THE PREPROCESSED DATA
# -------------------------------------------------------------------------------
# Generate a lists of all protein-ligand complexes, the corresponding folder path and protein_dictionary paths
input_data_dir = '/data/grbv/PDBbind/input_data_processed/'
complexes = [subfolder for subfolder in os.listdir(input_data_dir) if len(subfolder) ==4 and subfolder[0].isdigit()]
folder_paths = [os.path.join(input_data_dir, complex) for complex in complexes]
protein_paths = [os.path.join(folder_path, f'{complex}_protein_dict.pkl') for complex, folder_path in zip(complexes, folder_paths)]
ligand_paths = [os.path.join(folder_path, f'{complex}_ligand_san.sdf') for complex, folder_path in zip(complexes, folder_paths)]
affinity_dict = load_object('/home/grabeda2/DTI/data_PDBbind/v2020_general_affinity_dict.pkl')
# -------------------------------------------------------------------------------


# CHECK WHICH COMPLEXES ARE PART OF THE TEST DATASETS (CASF2013 and CASF2016)
# -------------------------------------------------------------------------------
casf_2013_dir = '/data/grbv/PDBbind/raw_data/CASF-2013/coreset'
casf_2016_dir = '/data/grbv/PDBbind/raw_data/CASF-2016/coreset'

casf_2013_complexes = [subfolder for subfolder in os.listdir(casf_2013_dir) if len(subfolder) ==4 and subfolder[0].isdigit()]
casf_2016_complexes = [subfolder for subfolder in os.listdir(casf_2016_dir) if len(subfolder) ==4 and subfolder[0].isdigit()]


# Are all 2013 complexes present in the preprocessed data? 
missing_2013 = []
for complex in casf_2013_complexes: 
    if not complex in complexes: missing_2013.append(complex)
print(f'CASF-2013 complexes that are not present in preprocessed data: {missing_2013}')

# Are all 2016 complexes present in the preprocessed data? 
missing_2016 = []
for complex in casf_2016_complexes: 
    if not complex in complexes: missing_2016.append(complex)
print(f'CASF-2016 complexes that are not present in preprocessed data: {missing_2016}')
# -------------------------------------------------------------------------------


# Create Output Folders
# -----------------------------------------------------------
train_folder = os.path.join(output_folder, 'training_data')
test_folder = os.path.join(output_folder, 'test_data')
casf2013_folder = os.path.join(test_folder, 'casf2013')
casf2016_folder = os.path.join(test_folder, 'casf2016')

for folder in [train_folder, test_folder, casf2013_folder, casf2016_folder]:
    if not os.path.exists(folder): os.makedirs(folder)
# -----------------------------------------------------------


# Initialize Log:
# -------------------------------------------------------------------------------
log_folder = output_folder + '.logs/'
if not os.path.exists(log_folder): os.makedirs(log_folder)
log_file_path = os.path.join(log_folder, "graph_generation.txt")
log = open(log_file_path, 'a')
log.write("Generation of Featurized Interaction Graphs - Log File:\n")
log.write("Data: PDBbind v2020 refined and general set merged\n")
log.write("\n")

skipped = []
num_threads = torch.get_num_threads() // 4
torch.set_num_threads(num_threads)
# -------------------------------------------------------------------------------



# Here start a loop over the mutants
#----------------------------------------------------------
# ind = complexes.index('1bcu')
# for complex_id, folder_path, protein_path, ligand_path in zip(complexes[ind:ind+10], folder_paths[ind:ind+10], protein_paths[ind:ind+10], ligand_paths[ind:ind+10]):

for complex_id, folder_path, protein_path, ligand_path in zip(complexes, folder_paths, protein_paths, ligand_paths):
    
    log_string = f'{complex_id}: '

    # Load necessary data
    protein_dict = load_object(protein_path)
    ligand = parse_sdf_file(ligand_path)
    connections_dict = load_object( os.path.join(folder_path, f'{complex_id}_connections.pkl') )
    esm_embedding = torch.load(os.path.join(folder_path, f'{complex_id}_{embedding_descriptor}.pt'))
    esm_emb_len = esm_embedding.shape[1]
    

    # Access the ligand mol object and generate coordinate matrix (pos)
    if not len(ligand) == 1:
        log_string += 'Skipped - More than one ligand molecule provided'
        log.write(log_string + "\n")
        skipped.append(complex_id)
        continue
    
    mol = ligand[0]
    conformer = mol.GetConformer()
    coordinates = conformer.GetPositions()
    pos = np.array(coordinates)



    #===================================================================================================================================
    # Create Interaction Graph
    #===================================================================================================================================


    # Edge Index and Node Feature Matrix for Substrate
    #------------------------------------------------------------------------------------------
    x = atom_features(mol, padding_len=esm_emb_len)
    lig_size = x.shape[0]

    if np.sum(np.isnan(x)) > 0:
        log_string += 'Skipped - Nans during ligand feature computation'
        log.write(log_string + "\n")
        skipped.append(complex_id)
        continue
    
    edge_index_lig, edge_attr_lig = edge_index_and_attr(mol, pos, self_loops=False, undirected=False)
    #------------------------------------------------------------------------------------------
    


    # Add the data of the amino acids identified as neighbors to X and POS
    #------------------------------------------------------------------------------------------
    connections = connections_dict['connections']
    connections_res_num = connections_dict['res_num']
    connections_res_name = connections_dict['res_name']

    # Joined Residues Dictionary
    protein = {}
    residue_idx = 1
    for chain in protein_dict:
        chain_comp = protein_dict[chain]['composition']

        if chain_comp == [True, False] or chain_comp == [True, True]:
            for residue in protein_dict[chain]['aa_residues']:
                protein[residue_idx] = protein_dict[chain]['aa_residues'][residue]
                residue_idx += 1

        elif chain_comp == [False, True]:
            for hetatm_res in protein_dict[chain]['hetatm_residues']:
                protein[residue_idx] = protein_dict[chain]['hetatm_residues'][hetatm_res]
                residue_idx += 1

    # Iterate over the connection enzyme residues and elongate X and POS
    new_indeces = []
    count = pos.shape[0]
    residue_mismatch = False
    incomplete_residue = False

    for residue, resname in zip(connections_res_num, connections_res_name):

        if not resname == protein[residue]['resname']:
            print(f'Complex {complex_id}: Residues do not match with')
            residue_mismatch = True
        
        # IF THE RESIDUE IS AN AMINO ACID
        if resname in amino_acids:
            
            try: ca_idx = protein[residue]['atoms'].index('CA')
            except ValueError as ve: 
                incomplete_residue = (residue, resname)
                continue
            
            # Add coords of the CA atom to pos
            coords = protein[residue]['coords'][ca_idx]
            pos = np.vstack((pos, coords))

            # Add feature vector (esm_embedding + padding) to x
            embedding = esm_embedding[residue-1]
            padding = np.array([0 for _ in range(num_atomfeatures)])
            features = np.concatenate((embedding, padding))
            x = np.vstack((x, features))

        # IF THE RESIDUE IS A HETATM
        else:
            # Add coords of hetatm to pos
            coords = protein[residue]['hetatmcoords']
            pos = np.vstack((pos, coords))

            # Add feature vector of hetatm to x
            resname_smiles = hetatm_smiles_dict1[resname.strip('0123456789')]
            hetatm_mol = Chem.MolFromSmiles(resname_smiles)
            features = atom_features(hetatm_mol, padding_len=esm_emb_len)
            x = np.vstack((x, features))

        new_indeces.append(count)
        count +=1
    #------------------------------------------------------------------------------------------



    # Check that no nans have been added to x during the feature computation
    if np.sum(np.isnan(x)) > 0:
        log_string += 'Skipped - Nans during enzyme residues feature computation'
        log.write(log_string + "\n")
        skipped.append(complex_id)
        continue

    # Check that there has been no residue mismatch
    if residue_mismatch: 
        log_string += 'Skipped - Mismatch between "connections" and protein_dict found!'
        log.write(log_string + "\n")
        skipped.append(complex_id)
        continue
    
    # If in one of the residues the CA atom was not found, PDB is incomplete, skip complex
    if incomplete_residue:
        log_string += f'Skipped - Protein residue {incomplete_residue} missing CA-Atom'
        log.write(log_string + "\n")
        skipped.append(complex_id)
        continue




    # EDGE INDEX, EDGE ATTR - Add the connection identified above to the edge_index
    #------------------------------------------------------------------------------------------

    mapping = {key: value for key, value in zip(connections_res_num, new_indeces)}

    edge_index_prot = [[],[]]
    edge_attr_prot = []

    for index, neighbor_list in enumerate(connections): 
        for enzyme_residue in neighbor_list:
            
            edge_index_prot[0]+=[index]
            edge_index_prot[1]+=[mapping[enzyme_residue]]

            distance = np.linalg.norm(pos[index]-pos[mapping[enzyme_residue]])

            # Add the feature vector of the new edges to new_edge_attr (2x)
            non_cov_feature_vec =   [0.,0.,1.,                # non-covalent interaction
                                    distance,                 # length 
                                    0.,0.,0.,0.,0.,           # bondtype = non-covalent
                                    0.,                       # is not conjugated
                                    0.,                       # is not in ring
                                    0.,0.,0.,0.,0.,0.]        # No stereo -> non-covalent

            # Add the feature vector of the new edges to new_edge_attr
            edge_attr_prot.append(non_cov_feature_vec)

    edge_index_prot = torch.tensor(edge_index_prot, dtype=torch.int64)
    edge_attr_prot = torch.tensor(edge_attr_prot, dtype=torch.float64)
    #------------------------------------------------------------------------------------------


    # Merging the two edge_indeces and edge_attrs
    edge_index = torch.concatenate( [edge_index_lig, edge_index_prot], axis=1 )
    edge_attr = torch.concatenate( [edge_attr_lig, edge_attr_prot], axis=0 )


    # Make undirected and add remaining self-loops
    edge_index, edge_attr = make_undirected_with_self_loops(edge_index, edge_attr)
    edge_index_prot, edge_attr_prot = make_undirected_with_self_loops(edge_index_prot, edge_attr_prot)
    edge_index_lig, edge_attr_lig = make_undirected_with_self_loops(edge_index_lig, edge_attr_lig)


    # Retrieve the binding affinity of the complex
    
    if 'Ki' in affinity_dict[complex_id].keys():
        affinity = affinity_dict[complex_id]['Ki']

    elif 'Kd' in affinity_dict[complex_id].keys():
        affinity = affinity_dict[complex_id]['Kd']

    elif 'IC50' in affinity_dict[complex_id].keys():
        affinity = affinity_dict[complex_id]['IC50']

    
    graph = Data(
        
            x = torch.tensor(x, dtype=torch.float64),
                 
            edge_index = edge_index,
            edge_index_lig = edge_index_lig,
            edge_index_prot = edge_index_prot,

            edge_attr = edge_attr,
            edge_attr_lig = edge_attr_lig,
            edge_attr_prot = edge_attr_prot,

            pos = torch.tensor(pos, dtype=torch.float64),
            affinity= torch.tensor(affinity, dtype=torch.float64),

            sizes = (lig_size, x.shape[0]-lig_size)
            )
    

    log_string += 'Successfully - Saved in '
    
    # Save the graph in test or training data folder: 
    # -------------------------------------------------------------------------------------------
    in_casf_2013 = False
    in_casf_2016 = False

    if complex_id in casf_2013_complexes:
        in_casf_2013 = True
        log_string += 'CASF2013 '
        torch.save(graph, os.path.join(casf2013_folder, f'{complex_id}_graph_{embedding_descriptor}.pt'))

    if complex_id in casf_2016_complexes:
        in_casf_2016 = True
        log_string += 'CASF2016 '
        torch.save(graph, os.path.join(casf2016_folder, f'{complex_id}_graph_{embedding_descriptor}.pt'))

    if (not in_casf_2013) and (not in_casf_2016):
        log_string += 'Training Data'
        torch.save(graph, os.path.join(train_folder, f'{complex_id}_graph_{embedding_descriptor}.pt'))


    log.write(log_string + "\n")

print(f'Graph Generation Finished - Skipped Complexes {skipped}')
log.close()