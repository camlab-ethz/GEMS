import argparse
from scipy.spatial import KDTree
import json
import sys
import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import warnings
import h5py
from scipy.spatial import KDTree
from time import time
from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Set up logging with rotation
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
log_file = 'pairwise_similarity_tm_rmsd_error_8.log'

# Set up a rotating file handler
rotating_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024, backupCount=3, encoding=None, delay=0)
rotating_handler.setFormatter(log_formatter)
rotating_handler.setLevel(logging.ERROR)

logger = logging.getLogger('root')
logger.setLevel(logging.ERROR)
logger.addHandler(rotating_handler)


def handle_signal(signal, frame):
    print("Signal received, closing files and exiting gracefully")
    sys.exit(0)


def parse_sdf_files(folder_path, complexes):
    parsed_molecules = {}
    for complex_id in complexes:
        sdf_path = os.path.join(folder_path, complex_id + '.sdf')
        try:
            mol = Chem.SDMolSupplier(sdf_path)[0]
            parsed_molecules[complex_id] = mol
        except:
            parsed_molecules[complex_id] = None
    return parsed_molecules



def parse_tm_align_output(output):
    tm_score1 = 0
    tm_score2 = 0
    seq_id = 0

    for line in output.split('\n'):
        if "TM-score=" in line and "normalized by length of Chain_1" in line:
            tm_score1 = float(line.split('=')[1].split()[0])
        if "TM-score=" in line and "normalized by length of Chain_2" in line:
            tm_score2 = float(line.split('=')[1].split()[0])
        if "Seq_ID=" in line:
            seq_id = float(line.split('=')[4])

    return max(tm_score1, tm_score2), seq_id



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



def run_tmalign(tm_align_path, pdb1, pdb2, matrix_path=None):

    if matrix_path:
        command = [tm_align_path, pdb1, pdb2, "-m", matrix_path]

    else: command = [tm_align_path, pdb1, pdb2]

    # Run the command
    result = subprocess.run(command, text=True, capture_output=True)

    if result.returncode == 0:
        #print("Output:\n", result.stdout)
        tm_score, seq_id = parse_tm_align_output(result.stdout)
        
        # Write result.stdout to txt
        # with open(matrix_path.replace('matrix', 'output'), 'w') as f:
        #     f.write(result.stdout)
        return tm_score, seq_id
    
    else:
        return None, None




def point_cloud_similarity_score(pc1, pc2):
    # Ensure both point clouds are numpy arrays
    pc1 = np.array(pc1)
    pc2 = np.array(pc2)
    
    # Build a KD-tree for the second point cloud
    tree = KDTree(pc2)
    
    # Find the nearest neighbors for each point in pc1
    distances, _ = tree.query(pc1)

    # Compute the root mean squared deviation
    rmsd = np.sqrt(np.mean(distances**2))
    
    return rmsd





def process_pair(id1, id2, mol1, mol2, folder_path, tm_align_path):
    
    matrix_path = os.path.join(folder_path, f"{id1}_{id2}_matrix.txt")

    tm_score = np.nan
    ligand_rmsd = np.nan
    try:

        # Align proteins with TM-align

        pdb1 = os.path.join(folder_path, f"{id1}.pdb")
        pdb2 = os.path.join(folder_path, f"{id2}.pdb")
        
        tm_score, seq_id = run_tmalign(tm_align_path, pdb1, pdb2, matrix_path)
        

        # Compute ligand positioning similarity
        rot_matrix, t_vector = parse_rotation_matrix_and_translation_vector(matrix_path)

        # Get the ligand atomcoords
        lig1_coords = mol1.GetConformer().GetPositions()
        lig2_coords = mol2.GetConformer().GetPositions()

        # Rotate and translate ligand1 to ligand2
        lig1_coords_moved = np.dot(lig1_coords, rot_matrix.T) + t_vector

        # Compute the positioning similarity
        ligand_rmsd = point_cloud_similarity_score(lig1_coords_moved, lig2_coords)

        if os.path.exists(matrix_path): os.remove(matrix_path)

    except Exception as e:
        logger.error(f"Error processing pair {id1}, {id2}: {str(e)}")

    return [tm_score, ligand_rmsd]







def main(folder_path, tm_align_path, start_complex=0, end_complex=19443):

    try:
        run = 8

        # List of the names of the complexes
        complexes = sorted([compl[0:4] for compl in os.listdir(folder_path) 
                            if compl[0].isdigit() and compl.endswith('.pdb')])

        num_complexes = len(complexes)
        print("Number of complexes: {}".format(num_complexes), flush=True)

        # Save list of complexes to json file
        with open('pairwise_similarity_complexes.json', 'w') as f:
            json.dump(complexes, f)
        print("List of complexes saved to pairwise_similarity_complexes.json", flush=True)


        # Initialize the HDF5 file and dataset to save the similarities
        with h5py.File(f'pairwise_similarity_tm_scores_{run}.hdf5', 'a') as f:
            if 'similarities' not in f:
                dset = f.create_dataset("similarities", (num_complexes, num_complexes), dtype='float32', compression="gzip")
            else:
                dset = f['similarities']


        with h5py.File(f'pairwise_similarity_rmsd_ligand_{run}.hdf5', 'a') as f:
            if 'similarities' not in f:
                dset = f.create_dataset("similarities", (num_complexes, num_complexes), dtype='float32', compression="gzip")
            else:
                dset = f['similarities']


        # Parse the SDF files and store the molecules in a dictionary
        print("Parsing all SDF files...", flush=True)
        parsed_molecules = parse_sdf_files(folder_path, complexes)
        print("Parsed all ligands!", flush=True)


        # Loop through each complex and compare it with all other complexes in parallel
        tic = time()
        for i in range(num_complexes):

            if not start_complex <= i < end_complex: continue

            to_compare = [j for j in range(i+1, num_complexes)]
            if len(to_compare) == 0: continue

            # RUN ALL THE COMPARISONS IN PARALLEL, accumulate the results
            results = Parallel(n_jobs=-1)(delayed(process_pair)(
                    complexes[i], complexes[j], parsed_molecules[complexes[i]], parsed_molecules[complexes[j]],
                    folder_path, tm_align_path) for j in to_compare)


            # APPEND THE TM SCORES TO THE HDF5 FILE
            with h5py.File(f'pairwise_similarity_tm_scores_{run}.hdf5', 'a') as f:
                dset = f['similarities']
                for j, metrics in zip(to_compare, results):
                    dset[i, j] = metrics[0]
                    dset[j, i] = metrics[0]

            # APPEND THE LIGAND RMSD TO THE HDF5 FILE
            with h5py.File(f'pairwise_similarity_rmsd_ligand_{run}.hdf5', 'a') as f:
                    dset = f['similarities']
                    for j, metrics in zip(to_compare, results):
                        dset[i, j] = metrics[1]
                        dset[j, i] = metrics[1]
            
            print(f"Time: {time() - tic:.2f} - Compared {complexes[i]} ({i}) to indexes {to_compare[0]}-{to_compare[-1]}", flush=True)


    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        sys.exit(1)

    toc = time()
    print("Elapsed time: {:.2f} seconds".format(toc - tic))




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Compute and store pairwise metrics for 3D complexes.")
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the 3D complexes')
    parser.add_argument('tm_align_path', type=str, help='Path to the TM-align executable')
    parser.add_argument('start_complex', type=int, default=0, help='Index of the first complex to process')
    parser.add_argument('end_complex', type=int, default=19443, help='Index of the last complex to process')
    args = parser.parse_args()

    main(args.folder_path, args.tm_align_path, args.start_complex, args.end_complex)

