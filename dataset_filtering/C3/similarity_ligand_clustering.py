import os
import pickle
import pandas as pd
import numpy as np
import csv
import time
from collections import Counter

from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Load all necessary data
smiles_dict = load_object('/data/grbv/PDBbind/DTI5_smiles_dict.pkl')
clustering_output = 'clusterRes_cluster_DTI5_1.tsv'
folder_path = '/data/grbv/PDBbind/DTI_5_c3'

casf2016_dir = '/data/grbv/PDBbind/DTI_5_c3/input_graphs_esm2_t6_8M/test_data/casf2016'
casf2013_dir = '/data/grbv/PDBbind/DTI_5_c3/input_graphs_esm2_t6_8M/test_data/casf2013'

casf2016_complexes = [filename[0:4] for filename in os.listdir(casf2016_dir) if 'graph' in filename]
casf2013_complexes = [filename[0:4] for filename in os.listdir(casf2013_dir) if 'graph' in filename]
test_complexes = casf2013_complexes + casf2016_complexes


def delete_complex(id, path):
    os.system(f'for file in {path}/*/training_data/{id}*.pt; do rm $file; done')


def analyse_cluster(name, data):
    
    N = len(data)

    distance_matrix = np.eye(N, N)
    conflicts = []
    removed = []
    reasons = []


    # CREATE FPSIM2 ENGINE OF THE CLUSTER TO ASSESS LIGAND SIMILARITIES
    #--------------------------------------------------------------------------------------------------
    db_file = f'/data/grbv/PDBbind/DTI_5_c3/clusters/{name}_db_smiles.h5'
    list_of_smiles = [[smiles_dict[compl[0]], idx ] for idx, compl in enumerate(data)]
    create_db_file(list_of_smiles, db_file, 'Morgan', {'radius': 2, 'nBits': 2048})

    try:
        fpe = FPSim2Engine(db_file)
    except:        

        # If FPSim2 Engine generation fails for a cluster, check if there is a test set complex
        # in the cluster and remove all training set complexes with similar affinity

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
        return name, removed, reasons

    

    # GENERATE DISTANCE MATRIX Iterate over the complexes in the cluster and compare each to all others 
    #----------------------------------------------------------------------------------------
    for i in range(N):

        query = smiles_dict[data[i][0]] # Get smiles of complex

        try:
            results = fpe.tversky(query, 0.4, 0.7, 0.3, n_workers=1)

            # Iterate over the detected similarities
            for col, value in results: 
                distance_matrix[i, col] = value
                
                # If the comparison is a value on the diagonal
                if col == i: continue

                # If the label is very different --> keep the pair
                affinity_difference = abs(float(data[i][2]) - float(data[col][2]))
                if affinity_difference > 1: continue

                # if both are in test sets, keep the pair
                if data[i][1].strip() == data[col][1].strip() == 'test': continue

                conflicts.append((data[i][0], data[col][0], round(value, 2), round(affinity_difference, 2)))


        except Exception as e:
            # Remove the datapoint for which Morgen Fingerprint fails
            delete_complex(data[i][0], folder_path)
            removed.append(data[i][0])
            reasons.append(e)
            continue
    

    distance_matrix = pd.DataFrame(distance_matrix, columns=[compl[0] for compl in data])
    distance_matrix.to_csv(f'/data/grbv/PDBbind/DTI_5_c2/clusters/{name}_distance_matrix.csv')

    # print('Before removing test similarities:')
    # print(conflicts)
    # print()

    # REMOVE SIMILARITIES TO THE TEST SET FROM THE TRAINING SET
    # ----------------------------------------------------------

    # Make a list of all conflicting complexes of the cluster
    conflicting = [word for tuple in conflicts for word in tuple if isinstance(word,str)]


    # REMOVE SIMILARITIES TO THE TEST SET FROM THE TRAINING DATASET
    # First check if any of them are in the test set, if yes, remove those they are similar to
    # ----------------------------------------------------------------------------------------

    for comp in conflicting:
        if comp in test_complexes:

            similarities = [tuple for tuple in conflicts if comp in tuple]
            to_remove = list(set([y for x, y, *_ in similarities if x == comp] + [x for x, y, *_ in similarities if y == comp]))
            
            for removal in to_remove:
                delete_complex(removal, folder_path)
                removed.append(removal)
                reasons.append([f'test dataset complex {comp}'])

                # Update the conflicts list
                conflicts = [tuple for tuple in conflicts if removal not in tuple]



    # REMOVE REMAINING REDUNDANCIES FROM THE TRAINING SET
    # ----------------------------------------------------------

    # A different threshold is used to remove similar training datapoints - Change conflicts list accordingly
    similarity_threshold = 0.4
    affinity_threshold = 1

    # print('After removing test similarities:')
    # print(conflicts)
    # print()

    conflicts = [tuple for tuple in conflicts if tuple[2] > similarity_threshold and tuple[3] < affinity_threshold]

    # print('Before removing training redundancy:')
    # print(conflicts)
    # print()

    while len(conflicts) > 0:

        # Recompile the list of all conflicting complexes of the cluster (with test-set-similarities removed
        # and with the new training similarities below the training similarity threshold removed)
        conflicting = [word for tuple in conflicts for word in tuple if isinstance(word,str)]
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
# Iterate over the clusters in CSV file

with open(clustering_output, 'r', newline='') as infile, open('dataset_cleaning_DTI5_c2.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    current_cluster = ''
    current = []

    for cluster_memb, id, dataset_memb, affinity in reader:

        new_cluster = cluster_memb != current_cluster
        
        if not new_cluster: 
            current.append([id, dataset_memb, affinity])
            
        elif new_cluster:# and current_cluster == '':

            # Do the analysis of the current cluster if it is has at least two members
            if len(current) > 1: 

                name, removed, reasons = analyse_cluster(current_cluster, current)
                writer.writerow([]) 

                print(name, removed)

                for r, reason in zip(removed, reasons):

                    if not isinstance(reason, list):
                        writer.writerow([f'{name} cluster: Removed {r} due to:\n {reason}']) 

                    else:
                        writer.writerow([f'{name} cluster: Removed {r} due to tversky similarity to {reason}'])     


            # Start a new cluster
            current = [[id, dataset_memb, affinity]]
            current_cluster = cluster_memb