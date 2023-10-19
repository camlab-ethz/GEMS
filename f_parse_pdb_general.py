from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
import numpy as np

def parse_pdb(parser, protein_id, filepath):
    
    ''' This function uses the PDBParser from BioPython to import a PDB file 
    and extract the chains, the residues, the atom types and atom coordinates of the protein. All data is saved in a nested
    dictionary and returned. The dictionary is built as follows: 

    protein = {0: { "aa_seq": "YH"
                    "chain_id": "A"
                    "coords": np.array([x,y,z],[x,y,z]...)
                    "residues": { 1: {'resname': 'TYR'
                                      'atom_indeces': [0,1,2,3,4, ...]
                                      'atoms: [CA, C, N, O, ...]
                                     }

                                  2: {'resname': 'HIS'
                                      'atom_indeces': [12,13,14,15,16, ...]
                                      'atoms: [CA, C, N, O, ...]
                                     }
                   }

               1: { "aa_seq": "VN"
                    "chain_id": "B"
                    "coords": np.array([x,y,z],[x,y,z]...)
                    "residues": { ... }
                   }    
              }

    Example: To get the residue name of the amino acid at position 179 of chain A:
        numpy_coords = protein['chain_A']['residues']['179']['resname'] 

    Example 2: To get theh AA-Sequence of chain B: 
        aa_seq = protein['chain_B']['aa_seq]
    
    '''

    structure = parser.get_structure(protein_id, filepath)

    atom_index = 0
    protein = {}

    # Loop over the chains and add the chains to the protein dictionary
    # ========================================================================================================================
    for j, chain in enumerate(structure.get_chains()):

        # Keep track what kind of residues there are in the chain - Amino Acids (aa), HETATM (het) and Water (wat)
        aa = False
        het = False

        aa_resnames = []
        aa_residues_dict = {}

        hetatm_residues_dict = {}
        water_residues_coords = []

        chain_atomcoords = []

        # Loop over the residues of the chain and collect data
        # ---------------------------------------------------------------------------
        for i, residue in enumerate(chain.get_residues()):
            
            res_id = residue.get_id()
            resname = residue.resname.strip()
            

            # Is the residue a non-water heteroatom?-------------
            if res_id[0].startswith("H") and resname != "HOH":
                het = True

                hetatmnames = []
                hetatm_coords = []

                for atom in residue.get_atoms():
                    hetatmnames.append(atom.get_name())
                    hetatm_coords.append(list(atom.get_vector()))

                hetatm_residues_dict[i] = {'resname':resname, 
                                            'atoms':hetatmnames,
                                            'hetatmcoords':np.array(hetatm_coords)}
            # ----------------------------------------------------
                           


            # Is the residue a water molecule?--------------------
            elif res_id[0].startswith("W") and resname == "HOH":

                for atom in residue.get_atoms():
                    water_residues_coords.append(list(atom.get_vector()))
            # ----------------------------------------------------
            
            
            
            # The residue is a amino acid----------------------------
            else:
                aa = True
                aa_resnames.append(resname)
                aa_residues_dict[i]={'resname':resname}
                
                # Loop over the atoms of the residue and collect data
                atoms = []
                atomnames = []
                residue_atomcoords = []
                # ----------------------------------------------------
                for atom in residue.get_atoms():
                    atoms.append(atom)

                    atomname = atom.get_name()
                    atomnames.append(atomname)
                    residue_atomcoords.append(list(atom.get_vector()))
                # ----------------------------------------------------

                # Keep track of the indeces of the atoms
                atom_indeces = [ind for ind in range(atom_index, atom_index + len(atoms))]
                atom_index += len(atoms)

                aa_residues_dict[i]['atom_indeces'] = atom_indeces
                aa_residues_dict[i]['atoms'] = atomnames
                aa_residues_dict[i]['coords'] = np.array(residue_atomcoords)

                #chain_atomcoords.extend(residue_atomcoords)
        # ---------------------------------------------------------------------------


        # Add the dictionary with data on the residues to the protein dict
        protein[j]={'aa_residues':aa_residues_dict}
        protein[j]['chain_id'] = chain.id
        #protein[j]['coords'] = np.array(chain_atomcoords)
        protein[j]['composition'] = [aa,het]
        protein[j]['hetatm_residues'] = hetatm_residues_dict
        protein[j]['water_residues'] = water_residues_coords

        # Save the aa_sequence
        aa_seq = seq1(''.join(aa_resnames))
        protein[j]['aa_seq'] = aa_seq

    # ========================================================================================================================

    return protein