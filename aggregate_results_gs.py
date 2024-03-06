import os
import csv
import argparse
import re

def parse_line(line):
    """Extract data from the line."""
    pattern = r"Epoch (\d+):.*Loss:\s+(\d+.\d+).*Pearson:\s+(\d+.\d+).*R2:\s+(\d+.\d+).*RMSE:\s+(\d+.\d+).*Loss:\s+(\d+.\d+).*Pearson:\s+(\d+.\d+).*R2:\s+(\d+.\d+).*RMSE:\s+(\d+.\d+)"
    match = re.search(pattern, line)
    return match.groups() if match else None


def find_pearson_coefficient(filepath):
    """
    Open the .out file and return the Validation Pearson Coefficient from the last "Saved" line.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if "Saved" in line:

                data = parse_line(line)
                

                if data:
                    # epochs.append(data[0])
                    # train_r.append(data[2])
                    # val_r.append(data[6])
                    # train_rmse.append(data[4])
                    # val_rmse.append(data[8])
                    return data[6]
                
                print(data)
                print('Coefficient Not Found') 
                break
    
    return None



def process_subfolders(root_path):
    """
    Traverse subfolders in the given path to find .out files and extract Pearson coefficients.
    """
    model_pearsons = {}

    runs = [os.path.join(root_path, dir) for dir in os.listdir(root_path) 
            if dir.startswith('gs_GAT0bn') and not dir.endswith('.tsv')]
    
    for run in sorted(runs):
        fold = os.path.join(run, 'Fold1')

        for filename in os.listdir(fold):
            
            if filename.endswith(".out"):
                filepath = os.path.join(fold, filename)
                pearson_coefficient = find_pearson_coefficient(filepath)
                if pearson_coefficient:
                    model_name = os.path.basename(run)
                    print(model_name, pearson_coefficient)
                    model_pearsons[model_name] = float(pearson_coefficient)

    return model_pearsons

def save_to_tsv(model_pearsons, output_tsv_path):
    """
    Save the model names and their corresponding Validation Pearson Coefficients to a TSV file.
    Sorted by the Pearson coefficient values.
    """
    # Sort the dictionary by Pearson coefficient values
    sorted_model_pearsons = sorted(model_pearsons.items(), key=lambda x: x[1], reverse=True)

    with open(output_tsv_path, 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['Model Name', 'Validation Pearson'])

        for model_name, pearson in sorted_model_pearsons:
            writer.writerow([model_name, pearson])



parser = argparse.ArgumentParser(description="Parse and write data from text files to a CSV.")
parser.add_argument("root_path", help="Directory path to search for files")
args = parser.parse_args()

# Specify the root directory and output CSV file path
root_path = args.root_path  # Update this path to your specific directory
output_tsv_path = os.path.join(args.root_path, 'gs_GAT0bn_f1_val_pearsons.tsv')

# Process the subfolders and save the results to a CSV file
model_pearsons = process_subfolders(root_path)
save_to_tsv(model_pearsons, output_tsv_path)
