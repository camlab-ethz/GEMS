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
log_file = 'pairwise_similarity_tanimoto_error.log'

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


def compute_tversky_similarity(mol1, mol2, alpha=0.5, beta=0.5):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # Generate fingerprints
    fp1 = mfpgen.GetCountFingerprint(mol1)
    fp2 = mfpgen.GetCountFingerprint(mol2)

    tversky_sim = DataStructs.TverskySimilarity(fp1, fp2, alpha, beta)
    return tversky_sim



def process_pair(id1, id2, mol1, mol2):
    
    try:
        # Compute Tanimoto/Tversky similarity
        if mol1 and mol2:
            tanimoto = compute_tversky_similarity(mol1, mol2)
            if tanimoto == 0: tanimoto = 0.001 # If twersky is zero, set it to a small value
        else:
            return 0

    except Exception as e:
        logger.error(f"Error processing pair {id1}, {id2}: {str(e)}")

    return tanimoto



def main(folder_path, save_as_json=False):

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


        # Initialize the HDF5 file and dataset to save the similarities
        with h5py.File('pairwise_similarity_tanimoto.hdf5', 'a') as f:
            if 'similarities' not in f:
                dset = f.create_dataset("similarities", (num_complexes, num_complexes), dtype='float32', compression="gzip")
            else:
                dset = f['similarities']


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

            print(f"Processing {complexes[i]} ({i})...", flush=True)


            # Check if the complex's similarities have already been precomputed
            # --------------------------------------------------------------------------------------------------------------
            precomputed_scores = os.path.join(folder_path, f"{complexes[i]}_similarities_tanimoto.json")
            
            if os.path.exists(precomputed_scores):
                
                try:
                    # Append tanimoto data to the h5py file
                    with open(precomputed_scores, 'r') as f:
                        scores = json.load(f)
                        scores = [scores[complex] for complex in complexes] # Convert values to list in the right order
                    with h5py.File('pairwise_similarity_tanimoto.hdf5', 'a') as f:
                        dset = f['similarities']
                        dset[i, :] = scores
                        dset[:, i] = scores


                    
                    print(f"--- Precomputed data found for {complexes[i]}, skipping...", flush=True)
                
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
                    print(f"--- Comparing {complexes[i]} ({i}) to indexes {to_compare[0]}-{to_compare[-1]}", flush=True)
                    results = Parallel(n_jobs=-1)(delayed(process_pair)(
                            complexes[i], complexes[j], parsed_molecules[complexes[i]], parsed_molecules[complexes[j]]) for j in to_compare)

                    # APPEND THE TANIMOTO SCORES TO THE HDF5 FILE
                    with h5py.File('pairwise_similarity_tanimoto.hdf5', 'a') as f:
                        dset = f['similarities']
                        for j, metrics in zip(to_compare, results):
                            dset[i, j] = metrics
                            dset[j, i] = metrics

                if save_as_json:

                    # Save the tanimoto scores to a JSON file
                    with h5py.File('pairwise_similarity_tanimoto.hdf5', 'a') as f:
                        dset = f['similarities']

                        line_data = np.array(dset[i, :], dtype=np.float32)
                        line_data = line_data.tolist()
                        sim_data = {complexes[j]: line_data[j] for j in range(num_complexes)}
                    
                    # Save the dictionary to a JSON file with indentation
                    with open(precomputed_scores, 'w') as f:
                        json.dump(sim_data, f, indent=4)
                        
                    print(f"--- Saved the tanimoto scores for {complexes[i]} to {precomputed_scores}", flush=True)

                toc = time()
                print(f"--- Done: Time needed: {toc - tac:.2f} seconds. Total Time: {toc - tic}", flush=True)
            # --------------------------------------------------------------------------------------------------------------


        # Make sure the HDF5 files are symmetric and the diagonal is correct
        with h5py.File(f'pairwise_similarity_rmsd_ligand.hdf5', 'a') as f:
            dset = f['similarities']
            dset[:] = np.maximum(dset[:], dset[:].T)

            arr = dset[:]
            np.fill_diagonal(arr, 0.0)
            dset[:] = arr

        with h5py.File(f'pairwise_similarity_tm_scores.hdf5', 'a') as f:
            dset = f['similarities']
            dset[:] = np.maximum(dset[:], dset[:].T)

            arr = dset[:]
            np.fill_diagonal(arr, 1.0)
            dset[:] = arr

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        sys.exit(1)

    toc = time()
    print(f"Elapsed time: {toc - tic:.2f} seconds", flush=True)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Compute and store pairwise metrics for 3D complexes.")
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the 3D complexes')
    parser.add_argument('--save_as_json', type=bool, default=False, help='Save the per complex similarities as JSON files')
    args = parser.parse_args()

    main(args.folder_path, save_as_json=args.save_as_json)

