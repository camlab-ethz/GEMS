import random
import argparse
from scipy.spatial import KDTree
import json
import sys
import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import warnings
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
log_file = f'pairwise_similarity_tm_rmsd_error.log'

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
        tm_score, seq_id = parse_tm_align_output(result.stdout)
        
        # Write result.stdout to txt
        # with open(matrix_path.replace('matrix', 'output'), 'w') as f:
        #     f.write(result.stdout)
        return tm_score, seq_id
    
    else:
        return None, None



def point_cloud_similarity_score(pc1, pc2):
    pc1 = np.array(pc1)
    pc2 = np.array(pc2)
    
    # Build KD-trees for both point clouds
    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)
    
    # Compute distances
    distances1, _ = tree2.query(pc1)
    distances2, _ = tree1.query(pc2)
    
    # Compute the symmetric root mean squared deviation using max
    return max(np.sqrt(np.mean(distances1**2)), np.sqrt(np.mean(distances2**2)))



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
        if os.path.exists(matrix_path): os.remove(matrix_path)
        logger.error(f"--- Error processing pair {id1}, {id2}: {str(e)}")

    return [tm_score, seq_id, ligand_rmsd]



def matrices_to_npy(similarity_matrix_tm, PSM_tm_scores_file,
                    similarity_matrix_rmsd, PSM_rmsd_file,
                    similarity_matrix_seqid, PSM_seqid_file):
        '''
        Saving matrices to the disk as .npy files, overwriting existing matrices. 
        Makes sure the np matrices are symmetric and the diagonal is correct.
        Convert matrices to float32 to save space.
        '''
        similarity_matrix_tm = np.maximum(similarity_matrix_tm[:], similarity_matrix_tm[:].T)
        np.fill_diagonal(similarity_matrix_tm, 1.0)
        np.save(PSM_tm_scores_file, similarity_matrix_tm.astype(np.float32))
        print(f"TM-score similarity matrix saved to {PSM_tm_scores_file}", flush=True)

        similarity_matrix_rmsd = np.minimum(similarity_matrix_rmsd[:], similarity_matrix_rmsd[:].T)
        np.fill_diagonal(similarity_matrix_rmsd, 0.0)
        np.save(PSM_rmsd_file, similarity_matrix_rmsd.astype(np.float32))
        print(f"RMSD similarity matrix saved to {PSM_rmsd_file}", flush=True)
        
        similarity_matrix_seqid = np.maximum(similarity_matrix_seqid[:], similarity_matrix_seqid[:].T)
        np.fill_diagonal(similarity_matrix_seqid, 1.0)
        np.save(PSM_seqid_file, similarity_matrix_seqid.astype(np.float32))
        print(f"Sequence identity similarity matrix saved to {PSM_seqid_file}", flush=True)



def main(folder_path, tm_align_path, save_as_json=False):

    try:
        # List of the names of the complexes
        complexes = sorted([compl[0:4] for compl in os.listdir(folder_path) 
                            if compl[0].isdigit() and compl.endswith('.pdb')])
        num_complexes = len(complexes)
        print("Number of complexes: {}".format(num_complexes), flush=True)

        # Save list of complexes to json file
        if not os.path.exists('pairwise_similarity_complexes.json'):
            with open('pairwise_similarity_complexes.json', 'w') as f:
                json.dump(complexes, f)
            print("List of complexes saved to pairwise_similarity_complexes.json", flush=True)


        # INITIALIZE PAIRWISE SIMILARITY MATRICES
        PSM_tm_scores_file = "pairwise_similarity_matrix_tm.npy"
        print("Initializing TM-score similarity matrix with zeros...", flush=True)
        similarity_matrix_tm = np.zeros((len(complexes), len(complexes)), dtype=np.float32)

        PSM_rmsd_file = "pairwise_similarity_matrix_rmsd.npy"
        print("Initializing RMSD similarity matrix with infinity...", flush=True)
        similarity_matrix_rmsd = np.ones((len(complexes), len(complexes)), dtype=np.float32) * np.inf

        PSM_seqid_file = "pairwise_similarity_matrix_seqid.npy"
        print("Initializing sequence identity similarity matrix with zeros...", flush=True)
        similarity_matrix_seqid = np.zeros((len(complexes), len(complexes)), dtype=np.float32)


        # Parse the SDF files and store the molecules in a dictionary
        print("Parsing all SDF files...", flush=True)
        parsed_molecules = parse_sdf_files(folder_path, complexes)
        print("Parsed all ligands!", flush=True)
        print()

        # --------------------------------------------------------------------------------------------------------------
        # Loop through each complex and compare it with all other complexes in parallel
        # --------------------------------------------------------------------------------------------------------------
        tic = time()
        for i in range(num_complexes):

            print(f"Processing {complexes[i]} ({i})...")


            # Check if the complex's similarities have already been precomputed
            # --------------------------------------------------------------------------------------------------------------
            precomputed_rmsds = os.path.join(folder_path, f"{complexes[i]}_similarities_rmsd.json")
            precomputed_tm_scores = os.path.join(folder_path, f"{complexes[i]}_similarities_tm_scores.json")
            precomputed_seqids = os.path.join(folder_path, f"{complexes[i]}_similarities_seqid.json")
            
            if os.path.exists(precomputed_rmsds) and os.path.exists(precomputed_tm_scores) and os.path.exists(precomputed_seqids):
                
                try:

                    # Append sequence identity data to the sequence identity matrix
                    with open(precomputed_seqids, 'r') as f:
                        seqid_data = json.load(f)
                        seqids = [seqid_data[complex] for complex in complexes]
                    similarity_matrix_seqid[i, :] = seqids
                    similarity_matrix_seqid[:, i] = seqids

                    # Append TM scores data to the TM matrix
                    with open(precomputed_tm_scores, 'r') as f:
                        tm_data = json.load(f)
                        tm_scores = [tm_data[complex] for complex in complexes]
                    similarity_matrix_tm[i, :] = tm_scores
                    similarity_matrix_tm[:, i] = tm_scores

                    # Append RMSD data to the RMSD matrix
                    with open(precomputed_rmsds, 'r') as f:
                        rmsd_data = json.load(f)
                        rmsds = [rmsd_data[complex] for complex in complexes]
                    similarity_matrix_rmsd[i, :] = rmsds
                    similarity_matrix_rmsd[:, i] = rmsds
                    
                    print(f"--- Precomputed data found for {complexes[i]}, skipping...")
                
                except Exception as e:
                    logger.error(f"Error in retrieving precomputed data: {str(e)}")
            # --------------------------------------------------------------------------------------------------------------



            # Compare complex to all other complexes in parallel
            # --------------------------------------------------------------------------------------------------------------
            else:
                to_compare = [j for j in range(i+1, num_complexes)]
                tac = time()
                if len(to_compare) > 0:

                    # RUN ALL THE COMPARISONS IN PARALLEL, accumulate the results
                    print(f"--- Comparing {complexes[i]} ({i}) to indexes {to_compare[0]}-{to_compare[-1]}")
                    results = Parallel(n_jobs=-1)(delayed(process_pair)(
                            complexes[i], complexes[j], parsed_molecules[complexes[i]], parsed_molecules[complexes[j]],
                            folder_path, tm_align_path) for j in to_compare)

                    # APPEND THE DATA TO THE SIMILARITY MATRICES
                    for j, metrics in zip(to_compare, results):
                        similarity_matrix_tm[i, j] = metrics[0]
                        similarity_matrix_tm[j, i] = metrics[0]

                        similarity_matrix_seqid[i, j] = metrics[1]
                        similarity_matrix_seqid[j, i] = metrics[1]                        

                        similarity_matrix_rmsd[i, j] = metrics[2]
                        similarity_matrix_rmsd[j, i] = metrics[2]

                if save_as_json:

                    # Save the TM scores to a JSON file
                    row = similarity_matrix_tm[i, :].tolist()
                    sim_data = {complexes[j]: row[j] for j in range(num_complexes)}
                    if not os.path.exists(precomputed_tm_scores):
                        with open(precomputed_tm_scores, 'w') as f:
                            json.dump(sim_data, f, indent=4)

                    # Save the SEQIDS to a JSON file
                    row = similarity_matrix_seqid[i, :].tolist()
                    sim_data = {complexes[j]: row[j] for j in range(num_complexes)}
                    if not os.path.exists(precomputed_seqids):
                        with open(precomputed_seqids, 'w') as f:
                            json.dump(sim_data, f, indent=4)

                    # Save the RMSD data to a JSON file
                    row = similarity_matrix_rmsd[i, :].tolist()
                    sim_data = {complexes[k]: row[k] for k in range(num_complexes)}
                    if not os.path.exists(precomputed_rmsds):
                        with open(precomputed_rmsds, 'w') as f:
                            json.dump(sim_data, f, indent=4)

                toc = time()
                print(f"--- Done: Time needed: {toc - tac:.2f} seconds. Total Time: {toc - tic}")

            # Save the similarity matrices to .npy files every 1000 complexes
            if (i + 1) % 1000 == 0: 
                matrices_to_npy(similarity_matrix_tm, PSM_tm_scores_file,
                                similarity_matrix_rmsd, PSM_rmsd_file,
                                similarity_matrix_seqid, PSM_seqid_file)
            # --------------------------------------------------------------------------------------------------------------


        # Save the final similarity matrices to .npy files
        matrices_to_npy(similarity_matrix_tm, PSM_tm_scores_file,
                        similarity_matrix_rmsd, PSM_rmsd_file,
                        similarity_matrix_seqid, PSM_seqid_file)


    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        sys.exit(1)

    toc = time()
    print(f"Elapsed time: {toc - tic:.2f} seconds", flush=True)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Compute and store pairwise metrics for 3D complexes.")
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the 3D complexes')
    parser.add_argument('tm_align_path', type=str, help='Path to the TM-align executable')
    parser.add_argument('--save_as_json', type=bool, default=False, help='Save the per complex similarities as JSON files')
    args = parser.parse_args()

    main(args.folder_path, args.tm_align_path, save_as_json=args.save_as_json)

