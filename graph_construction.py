import os
import glob
import argparse
import numpy as np

from Bio.PDB.PDBParser import PDBParser

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.MolStandardize import rdMolStandardize  

from time import time
# import jax.numpy as jnp
# from jax import jit, vmap

from f_parse_pdb_general import parse_pdb

# PyTorch and PyTorch Geometric
import torch
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.data import Data, Batch


def arg_parser():
    parser = argparse.ArgumentParser(description="Inputs to Graph Generation Script")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory containing all proteins(PDB) and ligands (SDF)')
    
    parser.add_argument('--replace', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], 
                    help="If existing graphs in the data_dir should be overwritten. Defaults to False.")
    
    parser.add_argument('--masternode',
                    default=False, type=lambda x: x.lower() in ['true', '1', 'yes'], 
                    help="If a masternode should be added to the graph. Defaults to False.")
    
    parser.add_argument('--protein_embeddings',
    nargs='+',
    help='Provide string to identify protein embeddings that should be incorporated (--protein embeddings string1 string2 string3).\
          The string should be a substring of the file names of the saved embeddings \
          (e.g. "*_esm2_t6_8M_UR50D" -> "esm2_t6_8M_URD50" or "esm2_t6")')

    parser.add_argument('--ligand_embeddings',
    nargs='+',
    help='Provide names of embeddings that should be incorporated (--ligand_embeddings string1 string2 string3).\
          The string should be a substring of the file names of the saved embeddings \
          (e.g. "*_ChemBERTa_10M_MLM" -> "ChemBERTa_10M_MLM" or "ChemBERTa_10M")')



    return parser.parse_args()



def parse_sdf_file(file_path):
    """
    Parses an SDF file and returns a list of molecules.

    Args:
        file_path (str): The path to the SDF file.

    Returns:
        list: A list of molecules parsed from the SDF file.
    """
    suppl = Chem.SDMolSupplier(file_path, sanitize=True, removeHs=True, strictParsing=True)
    molecules = []
    for mol in suppl:
        if mol is not None:
            molecules.append(mol)
    return molecules



def find_files_with_patterns(directory, pattern1, pattern2):
    """
    Find files in a directory that match the given patterns.

    Args:
        directory (str): The directory to search for files.
        pattern1 (str): The first pattern to match in the file names.
        pattern2 (str): The second pattern to match in the file names.

    Returns:
        list: A list of file paths that match the given patterns.
    """

    search_pattern = os.path.join(directory, f"*{pattern1}*{pattern2}*")
    matching_files = glob.glob(search_pattern)
    
    return matching_files



def one_of_k_encoding_unk(x, allowable_set):
    """
    Encodes a categorical variable using a one-hot encoding scheme. If the variable is not in the allowable set,
    it is encoded as the last element in the set.

    Parameters:
    x (str): The categorical variable to be encoded.
    allowable_set (list): The set of allowable categories for the variable.

    Returns:
    list: A list of boolean values indicating whether each category in the allowable set matches the input variable.
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def one_of_k_encoding(x, allowable_set):
    """
    Encodes a categorical variable using a one-hot encoding scheme.

    Parameters:
    x (str): The categorical variable to be encoded.
    allowable_set (list): The set of allowable categories for the variable.

    Returns:
    list: A list of boolean values indicating whether each category in the allowable set matches the input variable.
    """

    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))



def make_undirected_with_self_loops(edge_index, edge_attr, undirected=True, self_loops=True):
    """
    Converts the given edge_index and edge_attr to an undirected graph with self-loops.

    Args:
        edge_index (Tensor): The edge indices of the graph.
        edge_attr (Tensor): The edge attributes of the graph.
        undirected (bool, optional): Whether to convert the graph to undirected. Defaults to True.
        self_loops (bool, optional): Whether to add self-loops to the graph. Defaults to True.

    Returns:
        edge_index (Tensor): The updated edge indices of the graph.
        edge_attr (Tensor): The updated edge attributes of the graph.
    """
    self_loop_feature_vector = torch.tensor([0., 1., 0.,  # it's a self-loop
                                              0, 0, 0, 0,  # length is zero
                                              0., 0., 0., 0., 0.,  # bondtype = None
                                              0.,  # is not conjugated
                                              0.,  # is not in ring
                                              0., 0., 0., 0., 0., 0.],  # No stereo -> self-loop
                                              dtype=torch.float)
    if undirected: edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    if self_loops: edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self_loop_feature_vector)
    return edge_index, edge_attr



def get_atom_features(mol, all_atoms, padding_len=0):
    """
    Get the atom features for a given molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule object.
        padding_len (int): The length of padding to be added.

    Returns:
        np.array: An array containing the atom features.
    """
    x = []
    for atom in mol.GetAtoms():

        padding = [0 for n in range(padding_len)]
        symbol = atom.GetSymbol()

        if symbol in metals: symbol = 'metal'
        elif symbol in halogens: symbol = 'halogen'
        if symbol == 'H': continue #atom_encoding = [0 for i in all_atoms]
        
        else: atom_encoding = one_of_k_encoding(symbol, all_atoms)
        
        ringm = [atom.IsInRing()]
        hybr = atom.GetHybridization()
        charge = [float(atom.GetFormalCharge())] 
        aromatic = [atom.GetIsAromatic()]
        mass = [atom.GetMass()/100]
        numHs = atom.GetTotalNumHs()
        degree = atom.GetDegree()
        chirality = str(atom.GetChiralTag())

        results =   atom_encoding + \
                    ringm  + \
                    one_of_k_encoding_unk(hybr, [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP2D, 
                                             Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED]) + \
                    charge + \
                    aromatic + \
                    mass + \
                    one_of_k_encoding(numHs, [0, 1, 2, 3, 4]) + \
                    one_of_k_encoding_unk(degree,[0, 1, 2, 3, 4, 5, 6, 7, 8, 'OTHER']) + \
                    one_of_k_encoding_unk(chirality, ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'OTHER']) + \
                    padding    
        
        x.append(results)
    return np.array(x, dtype=np.float32)



def edge_index_and_attr(mol, pos, undirected = True, self_loops = True):
    """
    Compute the edge index and edge attributes for a given molecule and atom positions.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule object.
        pos (numpy.ndarray): The positions of atoms in the molecule.
        undirected (bool, optional): Whether to make the graph undirected. Defaults to True.
        self_loops (bool, optional): Whether to add self-loops to the graph. Defaults to True.

    Returns:
        torch.Tensor: The edge index tensor.
        torch.Tensor: The edge attribute tensor.
    """
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

        # Edge Type (covalent bond, non-covalent_bond, self-loop)
        edge_feature_vector.extend(one_of_k_encoding('covalent', ['covalent','self-loop','non-covalent']))

        # Length of Edge (append 4 times for compatibility with non-conv edge feature vectors)
        length = np.linalg.norm(pos[atm1]-pos[atm2])
        for i in range(4): edge_feature_vector.append(length/10)

        # Bond Type (single, double, aromatic)
        edge_feature_vector.extend(one_of_k_encoding(bond.GetBondTypeAsDouble(), [0.,1.0,1.5,2.0,3.0]))

        # Conjugated
        edge_feature_vector.append(bond.GetIsConjugated())

        # Is in Ring?
        edge_feature_vector.append(bond.IsInRing())

        # Stereo
        allowed = [Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOANY, 
                Chem.rdchem.BondStereo.STEREOE, 
                Chem.rdchem.BondStereo.STEREOZ, 
                Chem.rdchem.BondStereo.STEREOCIS, 
                Chem.rdchem.BondStereo.STEREOTRANS]
        
        edge_feature_vector.extend(one_of_k_encoding(bond.GetStereo(), allowed))

        edge_attr.append(edge_feature_vector)

    # Make undirected and add self loops if necessary
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_index, edge_attr = make_undirected_with_self_loops(edge_index, edge_attr, undirected=undirected, self_loops=self_loops)
    return edge_index, edge_attr


def calculate_cbeta_position(ca_coords, c_coords, n_coords):
    # Convert input coordinates to numpy arrays
    ca = np.array(ca_coords)
    c = np.array(c_coords)
    n = np.array(n_coords)
    
    # Bond lengths and angles
    bond_length_ca_cb = 1.54  # Å
    bond_angle_n_ca_cb = np.deg2rad(109.5)  # radians
    bond_angle_c_ca_cb = np.deg2rad(109.5)  # radians
    
    # Unit vectors along the bonds
    u_n_ca = (n - ca) / np.linalg.norm(n - ca)
    u_c_ca = (c - ca) / np.linalg.norm(c - ca)
    
    # Orthogonal vector to the plane formed by N, Cα, and C
    u_orth = np.cross(u_n_ca, u_c_ca)
    u_orth /= np.linalg.norm(u_orth)  # Normalize
    
    # Vector component in the plane
    u_plane = np.cross(u_orth, u_n_ca)
    u_plane /= np.linalg.norm(u_plane)  # Normalize
    
    # Compute the Cβ position
    cb = (  ca + bond_length_ca_cb 
            * (np.cos(bond_angle_n_ca_cb) 
            * u_n_ca 
            + np.sin(bond_angle_n_ca_cb) 
            * (np.cos(bond_angle_c_ca_cb) * u_plane + np.sin(bond_angle_c_ca_cb) * u_orth)
         ))
    
    return cb


class SkipComplexException(Exception):
    """Exception to end the current iteration and continue with the next complex."""
    pass

class FatalException(Exception):
    """Exception to end the script completely."""
    pass

# Parse the arguments
args = arg_parser()
data_dir = args.data_dir
replace_existing_graphs = args.replace
protein_embeddings = args.protein_embeddings
ligand_embeddings = args.ligand_embeddings
masternode = args.masternode


all_atoms = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'metal', 'halogen']
halogens = ['F', 'Cl', 'Br', 'I', 'At'] #Halogen atoms Fluorine (F), Chlorine (Cl), Bromine (Br), Iodine (I), and Astatine (At)
metals = [
    # Alkali Metals
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    # Alkaline Earth Metals
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    # Transition Metals
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv',
    # Lanthanides
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    # Actinides
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 
    'Es', 'Fm', 'Md', 'No', 'Lr',
    # Post-Transition Metals
    'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Nh', 'Fl', 'Mc', 'Lv',
    # Half-Metals
    'As', 'Si', 'Sb', 'Te'
]


amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
               "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

hetatm_smiles_dict = {'ZN': '[Zn+2]', 'MG': '[Mg+2]', 'NA': '[Na+1]', 'MN': '[Mn+2]', 'CA': '[Ca+2]', 'K': '[K+1]',
                      'NI': '[Ni+2]', 'FE': '[Fe+2]', 'CO': '[Co+2]', 'HG': '[Hg+2]', 'CD': '[Cd+2]', 'CU': '[Cu+2]', 
                      'CS': '[Cs+1]', 'AU': '[Au+1]', 'LI': '[Li+1]', 'GA': '[Ga+3]', 'IN': '[In+3]', 'BA': '[Ba+2]',
                      'RB': '[Rb+1]', 'SR': '[Sr+2]'}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Get sorted lists of proteins and ligands (dirEntry objects) in the data_dir
proteins = sorted([protein for protein in os.scandir(data_dir) if protein.name.endswith('protein.pdb')], key=lambda x: x.name)
ligands = sorted([ligand for ligand in os.scandir(data_dir) if ligand.name.endswith('ligand_san.sdf')], key=lambda x: x.name)

print("Construction of Featurized Interaction Graphs\n", flush=True)
print(f'Number of Protein PDBs: {len(proteins)}', flush=True)
print(f'Number of Ligand SDFs: {len(ligands)}', flush=True)

print(f'Protein Embeddings: {protein_embeddings}', flush=True)
print(f'Ligand Embeddings: {ligand_embeddings}', flush=True)

# Ensure that the lengths of proteins and ligands match
assert len(proteins) == len(ligands), "Proteins and ligands lists must be of the same length."
N = len(proteins)


# Initialize PDB Parser
parser = PDBParser(PERMISSIVE=1, QUIET=True)

amino_acids = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
known_hetatms = ['ZN','MG','NA','MN','CA','K','NI','FE','CO','HG','CD','CU','CS','AU','LI','GA','IN','BA','RB','SR']
known_residues = amino_acids + known_hetatms


# Start a loop over the complexes
#----------------------------------------------------------
for i, (protein, ligand) in enumerate(zip(proteins, ligands)):

    print(f'Processing Complex {protein.name} ({i+1}/{len(proteins)})', end=': ', flush=True)

    try:
        id = protein.name.split('_')[0]
        protein_path = protein.path
        ligand_path = ligand.path

        # Check if the graph of this complex exists already
        if not replace_existing_graphs and os.path.exists(os.path.join(data_dir, f'{id}_graph.pth')):
            raise SkipComplexException('Graph already exists')

        # If there is a mismatch between the protein and ligand names, skip the complex
        # (there should be a 1:1 correspondence between protein and ligand files in the data_dir)
        if protein.name.split('_')[0] != ligand.name.split('_')[0]:
            raise FatalException(f'Protein {protein.name} and Ligand {ligand.name} do not match')


        # LIGAND PARSING - Continue only if the parsed ligand has been processed successfully, else skip this complex
        lig = parse_sdf_file(ligand_path)
        if len(lig) == 1: ligand_mol = lig[0]
        else: raise SkipComplexException('Ligand could not be parsed successfully')

        
        # LIGAND ATOMCOORDS - Get coordinate Matrix of the ligand (Continue only if the ligand has at least 5 heavy atoms)
        conformer = ligand_mol.GetConformer()
        coordinates = conformer.GetPositions()
        ligand_atomcoords = np.array(coordinates, dtype=np.float32)
        if ligand_atomcoords.shape[0]<5: 
            raise SkipComplexException('Ligand is smaller than 5 Atoms and is skipped')


        # PARSING OF PROTEIN PDB TO GENERATE COORDINATE MATRIX
        # -----------------------------------------------------
        with open(protein_path) as pdbfile:
            protein_dict = parse_pdb(parser, id, pdbfile)

        # Iterate over the chains in the protein_dict and collect data on the amino acids and hetatms
        protein_atomcoords = np.array([], dtype=np.float32).reshape(0,3)
        res_list = []
        residue_memberships = []
        residues_dict = {} # Create a dictionary with all residues consecutively without chain distinction (for graph construction)

        residue_idx = 1
        for chain in protein_dict:
            chain_comp = protein_dict[chain]['composition']

            # CHAIN CONTAINS ONLY AMINO ACIDS AND HETATMS
            if chain_comp == [True, False] or chain_comp == [True, True]:
                
                for residue in protein_dict[chain]['aa_residues']:
                    res_dict = protein_dict[chain]['aa_residues'][residue]

                    # Add the residue to the residue_dict
                    residues_dict[residue_idx] = res_dict

                    # Process the information on the residue
                    protein_atomcoords = np.vstack((protein_atomcoords, res_dict['coords']))
                    res_list.append((residue_idx, res_dict['resname']))          
                    memb = [residue_idx for atom in res_dict['atom_indeces']]
                    residue_memberships.extend(memb)
                    residue_idx += 1
            
            # CHAIN CONTAINS HETATMS BUT NO AMINO ACIDS
            elif chain_comp == [False, True]: 
                for hetatm_res in protein_dict[chain]['hetatm_residues']:
                    hetatmres_dict = protein_dict[chain]['hetatm_residues'][hetatm_res]
                    
                    # Add the residue to the residue_dict
                    residues_dict[residue_idx] = hetatmres_dict
                    
                    # Process the information on the hetatm
                    protein_atomcoords = np.vstack((protein_atomcoords, hetatmres_dict['hetatmcoords']))
                    res_list.append((residue_idx, hetatmres_dict['resname']))
                    memb = [residue_idx for atom in hetatmres_dict['atoms']]
                    residue_memberships.extend(memb)
                    residue_idx +=1





        # # COMPUTE CONNECTIVITY BETWEEN LIGAND AND PROTEIN ATOMS
        # # -----------------------------------------------------
        
        # With numpy --------------------------------------------------------------------------------------------------------
        max_len = 4
        diff = protein_atomcoords[np.newaxis, :, :] - ligand_atomcoords[:, np.newaxis, :]
        pairwise_distances = np.linalg.norm(diff, axis=2)
        close = pairwise_distances <= max_len + 1
        
        connections = [np.unique(np.array(residue_memberships)[np.where(row)]) for row in close]
        connections_res_num = sorted(list(set([atm for l in connections for atm in l])))
        connections_res_name = [res_list[aa-1][1] for aa in connections_res_num]
        # ---------------------------------------------------------------------------------------------------------------------



        # With JAX --------------------------------------------------------------------------------------------------------
        #Warm up the JIT compilation

        # def compute_connections(protein_atomcoords, ligand_atomcoords):

        #     diff = protein_atomcoords[jnp.newaxis, :, :] - ligand_atomcoords[:, jnp.newaxis, :]
        #     pairwise_distances = jnp.linalg.norm(diff, axis=2)
        #     close = pairwise_distances <= 5
        #     return close
        # compute_connections = jit(compute_connections)

        # _ = compute_connections(np.array(protein_atomcoords), jnp.array(ligand_atomcoords))
        
        # tic = time()
        # close = compute_connections(protein_atomcoords, ligand_atomcoords)    
        # connections = [np.unique(np.array(residue_memberships)[np.where(row)]) for row in np.array(close)]
        # jnp_time = time()-tic
        # connected_res_num = sorted(list(set([atm for l in connections for atm in l])))
        # connected_res_name = [res_list[aa-1][1] for aa in connected_res_num]

        # print(f'JaxNumpy Time: {jnp_time}')
        # ---------------------------------------------------------------------------------------------------------------------



        unknown_res = [(res not in known_residues and res.strip('0123456789') not in known_residues) for res in connections_res_name]
        if any(unknown_res):
            raise SkipComplexException('Ligand has been connected to a unknown protein residue, the complex is therefore skipped')



        #-------------------------------------------------------------------------------------------------------------
        # GENERATE INTERACTION-GRAPH
        #------------------------------------------------------------------------------------------------------------- 

        num_atomfeatures = 40
        num_edgefeatures = 20


        # Load the amino acid embeddings
        if protein_embeddings:
            found_all_emb = True
            aa_embeddings = {}

            for j, emb in enumerate(protein_embeddings):
                matching_files = find_files_with_patterns(data_dir, id, emb)
                if len(matching_files) == 1:
                    aa_embeddings[j] = torch.load(matching_files[0])
                elif len(matching_files) == 0:
                    found_all_emb = False
                    break
                elif len(matching_files) > 1:
                    found_all_emb = False
                    break
                    
            # Skip the complex if not all/too many embeddings are found
            if not found_all_emb: 
                raise SkipComplexException(f'Not all protein embeddings found for {emb}')


        # Load the ligand embeddings if there are any
        if ligand_embeddings is not None:
            found_all_emb = True
            lig_embeddings = {}

            for j, emb in enumerate(ligand_embeddings):
                matching_files = find_files_with_patterns(data_dir, id, emb)
                if len(matching_files) == 1:
                    lig_embeddings[j] = torch.load(matching_files[0])
                elif len(matching_files) == 0:
                    found_all_emb = False
                    break
                elif len(matching_files) > 1:
                    found_all_emb = False
                    break
                    
            
            # Skip the complex if not all/too many embeddings are found
            if not found_all_emb:
                raise SkipComplexException('Not all ligand embeddings found')
        
        

        #------------------------------------------------------------------------------------------
        # Edge Index, Edge Attributes, Node Feature Matrix X and Coordinate Matrix POS for Ligand
        # - initialize node feature matrix X by computing the atom features for the ligand with RDKit
        # - write the edge index and edge attributes for the ligand with RDKit
        # - initialize the coordinate matrix POS with the ligand atom coordinates
        #------------------------------------------------------------------------------------------
        x = get_atom_features(ligand_mol, all_atoms, padding_len=len(amino_acids))
        
        if np.sum(np.isnan(x)) > 0:
            raise SkipComplexException('Nans during ligand feature computation')
        
        edge_index_lig, edge_attr_lig = edge_index_and_attr(ligand_mol, ligand_atomcoords, self_loops=False, undirected=False)
        
        # Initialize POS using the ligand coordinates
        pos = ligand_atomcoords.copy()




        #------------------------------------------------------------------------------------------
        # Extend Feature Matrix X and Coordinate Matrix POS with Protein Residues, initialize embedding feature matrix
        # - initialize embedding protein feature matrix
        # - iterate over the protein residues and add their coords to POS, their features to X, and their embeddings to X_EMB
        #------------------------------------------------------------------------------------------

        # Check if all imported amino embeddings have the same number of amino acids
        if protein_embeddings:
            num_AAs = [aa_embeddings[j].shape[0] for j, _ in enumerate(protein_embeddings)]
            if not all(len == num_AAs[0] for len in num_AAs):
                raise SkipComplexException('Embeddings have different lengths')

        
        # Initialize embedding protein feature matrix for each protein embedding
        # Add a row of zeros for each ligand atom (no embedding for ligand atoms)
        if protein_embeddings:
            x_emb = [np.zeros([x.shape[0], aa_embeddings[j].shape[1]], dtype=np.float32) for j,_ in enumerate(protein_embeddings)]

        
        # Iterate over the residues that were identified as neighbors of ligand atoms (<5A distance)
        # and add their coordinates to POS, their feature vectors to X, and their embeddings to X_EMB
        new_indeces = []
        count = pos.shape[0]

        for residue, resname in zip(connections_res_num, connections_res_name):

            if not resname == residues_dict[residue]['resname']:
                raise SkipComplexException(f'Residues in connection do not match with residues_dict')
    

            # IF THE RESIDUE IS AN AMINO ACID
            # -----------------------------------------------------
            if resname in amino_acids:

                try: 
                    ca_idx = residues_dict[residue]['atoms'].index('CA')
                    ca_coords = residues_dict[residue]['coords'][ca_idx]

                except ValueError as ve: 
                    raise SkipComplexException(f'Residue {residue, resname} is missing backbone atoms')

                # Add coords of the CA atom to pos
                pos = np.vstack((pos, ca_coords))

                # Add feature vector (one-hot-encoding of amino acid type) to feature matrix x
                aa_identity = np.array(one_of_k_encoding(resname, amino_acids))[np.newaxis,:]
                padding = np.zeros([1, num_atomfeatures], dtype=np.float32)
                features = np.hstack((padding, aa_identity))
                
                x = np.vstack((x, features))
                

                # For each embedding protein feature matrix, add the corresponding amino acid embedding
                if protein_embeddings:
                    for j,_ in enumerate(protein_embeddings):
                        aa_emb = aa_embeddings[j][residue-1]
                        
                        x_emb[j] = np.vstack((x_emb[j], aa_emb[np.newaxis,:]))
                    

            # IF THE RESIDUE IS A HETATM
            # -----------------------------------------------------
            else:
                # Add coords of hetatm to pos
                coords = residues_dict[residue]['hetatmcoords']
                pos = np.vstack((pos, coords))

                # Get the atom features for the heteroatom
                resname_smiles = hetatm_smiles_dict[resname.strip('0123456789')]
                hetatm_mol = Chem.MolFromSmiles(resname_smiles)
                hetatm_features = get_atom_features(hetatm_mol, all_atoms, padding_len=len(amino_acids))

                x = np.vstack((x, hetatm_features))

                # For each embedding protein feature matrix, add the a row of zeros
                if protein_embeddings:
                    for j,_ in enumerate(protein_embeddings):
                        padding = np.zeros([1, aa_embeddings[j].shape[1]], dtype=np.float32)

                        x_emb[j] = np.vstack((x_emb[j], padding))


            new_indeces.append(count)
            count +=1




        #------------------------------------------------------------------------------------------
        # EDGE INDEX, EDGE ATTR - Add the connection between ligand and protein nodes to the edge_index
        # - creates edge_index_prot (edges connecting ligand atoms to protein residues)
        # - creates edge_attr_prot (edge attributes for the edges connecting ligand atoms to protein residues)
        # - merges edge_index_lig and edge_index_prot into edge_index (edges connecting all nodes of the graph)
        #------------------------------------------------------------------------------------------

        # Create a mapping from the residue numbers in the protein to the indeces in the graph
        mapping = {key: value for key, value in zip(connections_res_num, new_indeces)}

        edge_index_prot = [[],[]]
        edge_attr_prot = []


        for index, neighbor_list in enumerate(connections): 
            for residue in neighbor_list:
                resname == residues_dict[residue]['resname']
                

                # --- EDGE INDEX ---
                edge_index_prot[0]+=[index]
                edge_index_prot[1]+=[mapping[residue]]

                # --- EDGE ATTR ---
                if resname in amino_acids:

                    # The connected protein residue is an amino acid - compute all distances between the ligand atom and 
                    # the four backbone atoms and construct a feature vector for the edge using the distances

                    try:
                        ca_idx = residues_dict[residue]['atoms'].index('CA')
                        c_idx = residues_dict[residue]['atoms'].index('C')
                        n_idx = residues_dict[residue]['atoms'].index('N')

                        ca_coords = residues_dict[residue]['coords'][ca_idx]
                        c_coords = residues_dict[residue]['coords'][c_idx]
                        n_coords = residues_dict[residue]['coords'][n_idx]
                        
                        cb_coords = calculate_cbeta_position(ca_coords, c_coords, n_coords)
                    
                    except ValueError as ve: 
                        raise SkipComplexException(f'Residue {residue, resname} is missing backbone atoms')

                    atm_ca = np.linalg.norm(pos[index] - ca_coords)
                    atm_n = np.linalg.norm(pos[index] - n_coords)
                    atm_c = np.linalg.norm(pos[index]- c_coords)
                    atm_cb = np.linalg.norm(pos[index] - cb_coords)

                    # Add the feature vector of the new edges to new_edge_attr (2x)
                    non_cov_feature_vec =   [0.,0.,1.,                                  # non-covalent interaction
                                            atm_ca/10, atm_n/10, atm_c/10, atm_cb/10,    # atm-bckbone distances divided by 10
                                            0.,0.,0.,0.,0.,                             # bondtype = non-covalent
                                            0.,                                         # is not conjugated
                                            0.,                                         # is not in ring
                                            0.,0.,0.,0.,0.,0.]                          # No stereo -> non-covalent

                else:

                    # The residue is a hetatm - compute the distance between the ligand atom and the hetatm
                    dist = np.linalg.norm(pos[index]-pos[mapping[residue]])

                    # Add the feature vector of the new edges to new_edge_attr (2x)
                    non_cov_feature_vec =   [0.,0.,1.,                                  # non-covalent interaction
                                            dist/10,dist/10,dist/10, dist/10,           # length divided by 10
                                            0.,0.,0.,0.,0.,                             # bondtype = non-covalent
                                            0.,                                         # is not conjugated
                                            0.,                                         # is not in ring
                                            0.,0.,0.,0.,0.,0.]                          # No stereo -> non-covalent
                
                # Add the feature vector of the new edges to new_edge_attr
                edge_attr_prot.append(non_cov_feature_vec)


        edge_index_prot = torch.tensor(edge_index_prot, dtype=torch.int64)
        edge_attr_prot = torch.tensor(edge_attr_prot, dtype=torch.float)

        # Merging the two edge_indeces and edge_attrs into an overall edge_index and edge_attr
        edge_index = torch.concatenate( [edge_index_lig, edge_index_prot], axis=1 )
        edge_attr = torch.concatenate( [edge_attr_lig, edge_attr_prot], axis=0 )

        # Make undirected and add remaining self-loops
        edge_index, edge_attr = make_undirected_with_self_loops(edge_index, edge_attr)
        edge_index_prot, edge_attr_prot = make_undirected_with_self_loops(edge_index_prot, edge_attr_prot)
        edge_index_lig, edge_attr_lig = make_undirected_with_self_loops(edge_index_lig, edge_attr_lig)
        #------------------------------------------------------------------------------------------



        #------------------------------------------------------------------------------------------
        # MASTER NODE 
        # - Edge Indeces: Write edge indeces to connect all nodes of the graph to a hypothetical master node
        # - Add a point with mean coordinates to the coordinate matrix
        # - Add a row of zeros to the feature matrices (standard x and protein embedding feature matrices x_emb)
        #------------------------------------------------------------------------------------------

        if masternode: 
            n_nodes = x.shape[0]
            n_l_nodes = ligand_atomcoords.shape[0]
            n_p_nodes = n_nodes - n_l_nodes

            # --- EDGE INDECES ---
            # For a masternode that is connected to all ligand atoms
            master_lig = [[i for i in range(n_l_nodes)]+[n_nodes],[n_nodes for _ in range(n_l_nodes+1)]]

            # For a masternode that is connected to all protein amino acids
            master_prot = [[i for i in range(n_l_nodes, n_nodes+1)],[n_nodes for _ in range(n_p_nodes+1)]]
            
            # For a masternode that is connected to all nodes of the graph
            edge_index_master_lig = torch.tensor(master_lig, dtype=torch.int64)
            edge_index_master_prot = torch.tensor(master_prot, dtype=torch.int64)
            edge_index_master = torch.concatenate( [edge_index_master_lig[:,:-1], edge_index_master_prot], dim=1)

            # Add a point with MEAN coordinates to the bottom of the coordinate matrix
            pos = np.vstack((pos, np.mean(pos, axis=0)))

            # Add a row of zeros to the feature matrix x
            x = np.vstack((x, np.zeros([1, x.shape[1]], dtype=np.float32)))

            # Add a row of zeros to the embedding feature matrices of the x_emb
            if protein_embeddings:
                for j,_ in enumerate(protein_embeddings):
                    x_emb[j] = np.vstack((x_emb[j], np.zeros([1, x_emb[j].shape[1]], dtype=np.float32)))

        #------------------------------------------------------------------------------------------



        #------------------------------------------------------------------------------------------
        # Check the shapes of the input tensors
        #------------------------------------------------------------------------------------------

        consistent = x.shape[0] == pos.shape[0]
        if consistent: N = x.shape[0]
        else: raise SkipComplexException(f'Inconsistent shapes of x and pos: {x.shape, pos.shape}')

        if pos.shape[1] != 3: raise SkipComplexException('POS has wrong shape')
        if x.shape[1] != num_atomfeatures + len(amino_acids): raise SkipComplexException(f'X has wrong shape {x.shape}')

        if not edge_index.max().item() < N: 
            raise SkipComplexException(f'Edge index out of bounds: {edge_index.max().item()} {N}')
        if not edge_index_lig.max().item() < N:
            raise SkipComplexException(f'Edge index lig out of bounds: {edge_index_lig.max().item()} {N}')
        if not edge_index_prot.max().item() < N:
            raise SkipComplexException(f'Edge index prot out of bounds: {edge_index_prot.max().item()} {N}')
        if not edge_index_master.max().item() < N:
            raise SkipComplexException(f'Edge index master out of bounds: {edge_index_master.max().item()} {N}')
        if not edge_index_master_lig.max().item() < N:
            raise SkipComplexException(f'Edge index master lig out of bounds: {edge_index_master_lig.max().item()} {N}')
        if not edge_index_master_prot.max().item() < N:
            raise SkipComplexException(f'Edge index master prot out of bounds: {edge_index_master_prot.max().item()} {N}')
        

        if protein_embeddings:
            for j, emb in enumerate(protein_embeddings):
                if x.shape[0] != x_emb[j].shape[0]:
                    raise SkipComplexException(f'Dimension 0 of x {x.shape} and {emb} {x_emb[j].shape} not identical ')
        
        for edge_ind, edge_at in [(edge_index.shape, edge_attr.shape),(edge_index_lig.shape, edge_attr_lig.shape),(edge_index_prot.shape, edge_attr_prot.shape)]:
            if edge_ind[0] != 2 or edge_at[1] != num_edgefeatures or edge_ind[1]!=edge_at[0]:
                raise SkipComplexException(f'Skipped - edge indeces shape error: \
                        {edge_index.shape, edge_attr.shape}\n\
                        {edge_index_lig.shape, edge_attr_lig.shape}\n\
                        {edge_index_prot.shape, edge_attr_prot.shape}')

        #------------------------------------------------------------------------------------------

        # Save the graph data dictionary
        graph = Data(
            
                x= torch.tensor(x, dtype=torch.float),

                edge_index= edge_index,
                edge_index_lig= edge_index_lig,
                edge_index_prot= edge_index_prot,

                edge_attr= edge_attr.float(),
                edge_attr_lig= edge_attr_lig.float(),
                edge_attr_prot= edge_attr_prot.float(),
                
                pos= torch.tensor(pos, dtype=torch.float),
                id= id,
        )

        if masternode: 
            graph.edge_index_master_lig = edge_index_master_lig
            graph.edge_index_master_prot = edge_index_master_prot
            graph.edge_index_master = edge_index_master
        
        # Add the amino acid embeddings to the graph_data_dict
        if protein_embeddings:
            #graph.protein_embeddings = protein_embeddings
            for j, emb_name in enumerate(protein_embeddings):
                graph[emb_name] = torch.tensor(x_emb[j], dtype=torch.float)

        
        # Add the ligand embeddings to the graph_data_dict
        if ligand_embeddings:
            for j, emb_name in enumerate(ligand_embeddings):
                graph[emb_name] = lig_embeddings[j].float()
        
        
        # Save the dictionary of graph data using torch.save
        torch.save(graph, os.path.join(data_dir, f'{id}_graph.pth'))
        print('Successful - Graph Saved', flush=True)


    # If an error occurs somewhere within the script, continue with the next complex
    except SkipComplexException as e:
        print('Error: ' + str(e), flush=True)
        continue

    # If a fatal error occurs, break the loop to end the script
    except FatalException as e:

        print(f"Fatal error: {e}", flush=True)
        break

    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)
        continue
