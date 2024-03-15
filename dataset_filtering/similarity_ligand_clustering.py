''' This version of the similarity ligand clustering should
- Only remove the redundancy in the training dataset
- Ignore all overlaps between test and train datasets
'''


import os
import pickle
import pandas as pd
import numpy as np
import csv
import argparse
from collections import Counter

from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



parser = argparse.ArgumentParser(description="Input Parameters for Dataset Filtering")
parser.add_argument("--folder_path", type=str, help="Directory with subfolders containing training data")
parser.add_argument("--seq_clusters", help="Path to CSV file generated during protein sequence clustering")
parser.add_argument("--similarity_threshold", default=0.4, type=float, help="Tversky Similarity Threshold for Train-Test Pairs")
parser.add_argument("--affinity_threshold", default=1.0, type=float, help="Affinity Threshold for Train-Test Pairs")
parser.add_argument("--similarity_threshold_train", default=0.4, type=float, help="Tversky Similarity Threshold for Train-Train Pairs")
parser.add_argument("--affinity_threshold_train", default=1.0, type=float, help="Affinity Threshold for Train-Test Pairs")

parser.add_argument("--rm_train_test_sims", default=True, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not the filtering should remove train-test set similarities")
parser.add_argument("--rm_train_train_sims", default=True, type=lambda x: x.lower() in ['true', '1', 'yes'], help="Wheter or not filtering should remove training set redundancies")

args = parser.parse_args()


# General Parameters for filtering
rm_train_test_sims = args.rm_train_test_sims
rm_train_train_sims = args.rm_train_train_sims

if not (rm_train_train_sims or rm_train_test_sims):
    raise Exception('Either "rm_train_train_sims" or "rm_train_test_sims" must be set to True')

# Paths to all necessary data
smiles_dict = load_object('/data/grbv/PDBbind/DTI5_smiles_dict.pkl')
seq_clustering_output = args.seq_clusters #'clusterRes_cluster_DTI5_1.tsv'
folder_path = args.folder_path
output_summary_csv = f'dataset_cleaning_logs.csv'

# Generate lists of complex ids in the test dataset and in the refined dataset
casf2016_dir = '/data/grbv/PDBbind/raw_data/CASF-2013/coreset'
casf2013_dir = '/data/grbv/PDBbind/raw_data/CASF-2016/coreset'
refined_dir = '/data/grbv/PDBbind/raw_data/v2020_refined'

casf2016_complexes = [folder for folder in os.listdir(casf2016_dir)]
casf2013_complexes = [folder for folder in os.listdir(casf2013_dir)]
refined_complexes = [folder for folder in os.listdir(refined_dir)]
test_complexes = casf2013_complexes + casf2016_complexes


# Similarity threshold for pair between train and test set 
similarity_threshold = args.similarity_threshold
affinity_threshold = args.affinity_threshold

#If a different threshold is used to remove similarities within the train set - Similarity threshold for pair between train and test set 
similarity_threshold_train = args.similarity_threshold_train
affinity_threshold_train = args.affinity_threshold_train


# Function to delete a complex from all datasets by its ID
def delete_complex(id, path):
    os.system(f'for file in {path}/*/training_data/{id}*.pt; do rm $file; done')



def analyse_cluster(name, data):
    
    '''Main function to analyse a cluster and determine which complexes should be removed from the dataset'''

    N = len(data)

    distance_matrix = np.eye(N, N)
    conflicts = []
    removed = []
    reasons = []


    # CREATE FPSIM2 ENGINE OF THE CLUSTER TO ASSESS LIGAND SIMILARITIES
    #--------------------------------------------------------------------------------------------------
    if not os.path.exists(f'{folder_path}/clusters/'): os.system(f'mkdir {folder_path}/clusters/')
    
    db_file = f'{folder_path}/clusters/{name}_db_smiles.h5'
    list_of_smiles = [[smiles_dict[compl[0]], idx ] for idx, compl in enumerate(data)]
    create_db_file(list_of_smiles, db_file, 'Morgan', {'radius': 2, 'nBits': 2048})

    #fpe = FPSim2Engine(db_file)

    try:
        fpe = FPSim2Engine(db_file)
    except: 
        # If FPSim2 Engine generation fails for a cluster     

        
        if rm_train_test_sims == True:
        # check if there is a test set complex in the cluster and if yes, remove all training set complexes with similar affinity
            
            while len(data) > 0:
                datasets = [d[1] for d in data]
                if 'test' in datasets:

                    for j, d in enumerate(data): 
                        if d[1] == 'test': break

                    test_complex = data[j][0]
                    test_affinity = float(data[j][2])
                    del data[j]
                    
                    delete = []
                    for d in data:
                        if d[1] == 'test': continue
                        if abs(float(d[2]) - test_affinity) < 1:
                            delete_complex(d[0], folder_path)
                            removed.append(d[0])
                            reasons.append([f'Test Dataset Complex {test_complex} Affinity (Parsing Fail)'])
                            delete.append(d)
                    
                    for trash in delete: data.remove(trash)
                else:
                    for d in data: data.remove(d)
                

        
        if rm_train_train_sims == True:
        # remove all training set complexes with similar affinity from the cluster
            
            # Step 0: Remove all test complexes from the list - we ignore those (or they where already handled above)
            data = [d for d in data if d[1] != 'test']

            # Step 1: Sort the list by affinity value
            sorted_data = sorted(data, key=lambda x: x[2])

            # Step 2: Filter the list to enforce the affinity difference > 1
            filtered_data = []

            for d in sorted_data:
                if not filtered_data or abs(d[2] - filtered_data[-1][2]) > 1:
                    filtered_data.append(d)
                else:
                    delete_complex(d[0], folder_path)
                    removed.append(d[0])
                    reasons.append([f'Training Dataset Complex {filtered_data[-1][0]} (Parsing Fail)'])

        
        return name, removed, reasons


    

    # GENERATE DISTANCE MATRIX 
    # - Iterate over the complexes in the cluster and compare each to all others
    # - Fill distance matrix and save it
    # - Save conflicts list of similar complexes
    #----------------------------------------------------------------------------------------
    for i in range(N):

        query = smiles_dict[data[i][0]] # Get smiles of complex

        try:
            results = fpe.tversky(query, similarity_threshold, 0.7, 0.3, n_workers=1)

            # Iterate over the detected similarities
            for col, value in results: 
                distance_matrix[i, col] = value
                
                # If the comparison is a value on the diagonal
                if col == i: continue

                # If the label is very different --> keep the pair
                affinity_difference = abs(float(data[i][2]) - float(data[col][2]))
                if affinity_difference > affinity_threshold: continue

                # if both are in test sets, keep the pair
                if data[i][1].strip() == data[col][1].strip() == 'test': continue

                conflicts.append((data[i][0], data[col][0], round(value, 2), round(affinity_difference, 2)))


        except Exception as e:
            # Remove the datapoint for which Morgen Fingerprint fails
            delete_complex(data[i][0], folder_path)
            removed.append(data[i][0])
            reasons.append(e)
            continue
    
    names = [compl[0] if compl[0] not in test_complexes else f'{compl[0]} (test)' for compl in data]
    distance_matrix = pd.DataFrame(distance_matrix, columns=names)
    distance_matrix.index = names
    distance_matrix.to_csv(f'{folder_path}/clusters/{name}_distance_matrix.csv')



    # REMOVE SIMILARITIES TO THE TEST SET FROM THE TRAINING SET
    # ----------------------------------------------------------

    # Make a list of all conflicting complexes of the cluster
    conflicting = [word for tuple in conflicts for word in tuple if isinstance(word,str)]


    # First check if any of them are in the test set, if yes, remove those they are similar to
    for comp in conflicting:
        if comp in test_complexes:


            # If removal of train-test similarities is turned ON
            # --------------------------------------------------
            if rm_train_test_sims == True:

                similarities = [tuple for tuple in conflicts if comp in tuple]
                to_remove = list(set([y for x, y, *_ in similarities if x == comp] + [x for x, y, *_ in similarities if y == comp]))
                
                for removal in to_remove:
                    delete_complex(removal, folder_path)
                    removed.append(removal)
                    reasons.append([f'test dataset complex {comp}'])

                    # Update the conflicts list
                    conflicts = [tuple for tuple in conflicts if removal not in tuple]
            # --------------------------------------------------


            # If removal of train-test similarities is turned OFF
            else: conflicts = [tuple for tuple in conflicts if comp not in tuple]





    # REMOVE REMAINING REDUNDANCIES FROM THE TRAINING SET
    # ----------------------------------------------------------         

    if rm_train_train_sims == True:

        conflicts = [tuple for tuple in conflicts if tuple[2] > similarity_threshold_train and tuple[3] < affinity_threshold_train]

        while len(conflicts) > 0:

            # Recompile the list of all conflicting complexes of the cluster (with test-set-similarities removed)
            conflicting = [word for tuple in conflicts for word in tuple if isinstance(word,str)]
            
            # Remove complexes of the general set with higher priority (because they are lower quality)
            conflicting_general = [compl for compl in conflicting if compl not in refined_complexes]
            
            if len(conflicting_general) > 0:
                word_count = Counter(conflicting_general)
                most_frequent,_ = word_count.most_common(1)[0]

            # If all general complexes are removed, remove also refined complexes
            else:
                word_count = Counter(conflicting)
                most_frequent,_ = word_count.most_common(1)[0]

            # Remove the most frequent complex
            delete_complex(most_frequent, folder_path)
            removed.append(most_frequent)
            similarities = [tuple for tuple in conflicts if most_frequent in tuple]
            reasons.append(list(set([y for x, y, *_ in similarities if x == most_frequent] + [x for x, y, *_ in similarities if y == most_frequent])))
            
            # Update the conflicts list
            conflicts = [tuple for tuple in conflicts if most_frequent not in tuple]
        
    return name, removed, reasons




# M A I N ----------------------------------------------------------------------------------------------------------------
# Iterate over the clusters in CSV file generated during protein sequence clustering


# Open input file (protein sequence clustering output) and output file for summary/logs
with open(seq_clustering_output, 'r', newline='') as infile, open(output_summary_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    current_cluster = ''
    current = []

    # Iterate over the lines of the sequence clustering output
    for cluster_memb, id, dataset_memb, affinity in reader:

        new_cluster = cluster_memb != current_cluster
        
        if not new_cluster: 
            current.append([id, dataset_memb, affinity])
            
        elif new_cluster:


            # Run the analysis of the current cluster if it is has at least two members
            # ------------------------------------------------------------------------
            if len(current) > 1: 

                name, removed, reasons = analyse_cluster(current_cluster, current)
                writer.writerow([]) 

                print(name, removed)

                for r, reason in zip(removed, reasons):

                    if r in refined_complexes: 
                        memb = "(refined)"
                    else: 
                        memb = "(general)"

                    if not isinstance(reason, list):
                        writer.writerow([f'{name} cluster: Removed {r} {memb} due to:\n {reason}']) 

                    else:
                        writer.writerow([f'{name} cluster: Removed {r} {memb} due to tversky similarity to {reason}'])     
            # ------------------------------------------------------------------------


            # Start a new cluster
            current = [[id, dataset_memb, affinity]]
            current_cluster = cluster_memb