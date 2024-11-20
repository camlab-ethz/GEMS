import os
import numpy as np
from Bio import PDB



def parse_rotation_matrix_and_translation_vector(filename):
    """
    Parses the rotation matrix and translation vector from a given text file.
    
    Args:
    filename (str): The path to the text file containing the matrix and vector.
    
    Returns:
    tuple: A tuple containing:
        - np.array: Rotation matrix (3x3)
        - np.array: Translation vector (1x3)
    """
    rotation_matrix = []
    translation_vector = []
    
    # Open and read the file
    with open(filename, 'r') as file:
        start_parsing = False
        for line in file:
            # Check if we've reached the matrix section
            if "------ The rotation matrix to rotate Chain_1 to Chain_2 ------" in line:
                start_parsing = True
                continue
            
            # Parse the matrix and vector
            if start_parsing:
                if line.strip() and 'm' not in line:  # Ensure it's not a header or empty line
                    parts = line.split()
                    translation_vector.append(float(parts[1]))
                    rotation_matrix.append([float(part) for part in parts[2:]])
                
                # Stop parsing after reading the third line of the matrix
                if len(rotation_matrix) == 3:
                    break
    
    return np.array(rotation_matrix), np.array(translation_vector)



def transform_structure(pdb_file, rotation_matrix, translation_vector, output_file):
    """
    Transforms the coordinates of atoms in a PDB file using a rotation matrix and translation vector.
    
    Args:
    pdb_file (str): Path to the PDB file to transform.
    rotation_matrix (np.array): 3x3 rotation matrix.
    translation_vector (np.array): 1x3 translation vector.
    output_file (str): Path to save the transformed PDB file.
    """
    # Parse the PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('chain1', pdb_file)
    
    # Apply the transformation to each atom in the structure
    for atom in structure.get_atoms():
        coord = atom.get_coord()
        # Apply rotation and translation
        new_coord = np.dot(rotation_matrix, coord) + translation_vector
        atom.set_coord(new_coord)
    
    # Save the transformed structure
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)



if __name__ == "__main__":
    import argparse
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Transform a PDB structure using a rotation matrix and translation vector.")
    parser.add_argument("pdb_file", help="Path to the input PDB file of chain 1.")
    parser.add_argument("matrix_file", help="Path to the text file containing the rotation matrix and translation vector.")
    parser.add_argument("output_file", help="Path to save the transformed PDB file.")
    
    args = parser.parse_args()
    
    # Parse the rotation matrix and translation vector
    rotation_matrix, translation_vector = parse_rotation_matrix_and_translation_vector(args.matrix_file)
    
    # Copy the matrix file to main
    os.system(f"cp {args.matrix_file} {os.path.dirname(args.output_file)}")
    
    # Transform the structure
    transform_structure(args.pdb_file, rotation_matrix, translation_vector, args.output_file)


# python rotate_pdb.py PDBbind_v2021/4eor_protein.pdb find_most_similar_after/4eor_to_5wij_matrix.txt 4eo3_on_5wij.pdb