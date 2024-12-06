#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
import csv
import json

def convert_csv_to_json(input_file, output_file):
    """
    Convert a CSV file to a JSON file with a specific structure.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output JSON file
    
    Returns:
        str: Path to the generated JSON file
    """
    # Read CSV data and convert to the desired JSON structure
    data_dict = {}
    try:
        with open(input_file, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";")
            for row in csv_reader:
                if len(row) == 2:  # Ensure there are exactly two items: key and value
                    key = row[0].strip()  # The outer key
                    value = row[1].strip().lstrip("_")  # Remove leading underscores
                    try:
                        # Build the required nested structure
                        data_dict[key] = {
                            "log_kd_ki": float(value),
                            "dataset": ["general"],  # Default dataset list
                        }
                    except ValueError:
                        # Skip non-numeric rows
                        print(f"Skipping row: {row}")
        
        # Write dictionary to JSON file
        with open(output_file, "w") as json_file:
            json.dump(data_dict, json_file, indent=4)
        
        print(f"Data successfully written to {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        sys.exit(1)



def run_command(command):
    """
    Run a shell command and handle potential errors.
    
    Args:
        command (list): List of command and its arguments
    
    Raises:
        subprocess.CalledProcessError: If the command fails
    """
    print(f"Executing: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)



def main():
    parser = argparse.ArgumentParser(description="Machine Learning Workflow Execution")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory for input and output data')
    parser.add_argument('--y_data', type=str, default=None, 
                        help='Optional path to CSV or JSON file containing y data')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        sys.exit(1)


    workflow_commands = [
        # Ankh Features
        ["python", "-m", "dataprep.ankh_features", 
         "--data_dir", args.data_dir, 
         "--ankh_base", "True"],
        
        # ESM Features
        ["python", "-m", "dataprep.esm_features", 
         "--data_dir", args.data_dir, 
         "--esm_checkpoint", "t6"],

        # ChemBerta Features
        ["python", "-m", "dataprep.chemberta_features", 
         "--data_dir", args.data_dir],
        
        # Graph Construction
        ["python", "-m", "dataprep.graph_construction", 
         "--data_dir", args.data_dir, 
         "--replace", "False", 
         "--protein_embeddings", "ankh_base", "esm2_t6", 
         "--ligand_embeddings", "ChemBERTa_77M"]
    ]
    
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))

    # Dataset Construction (conditionally add y_data)
    dataset_command = ["python", "-m", "dataprep.construct_dataset", 
                       "--data_dir", args.data_dir, 
                       "--protein_embeddings", "ankh_base", "esm2_t6", 
                       "--ligand_embeddings", "ChemBERTa_77M", 
                       "--save_path", f"{os.path.dirname(args.data_dir)}{data_dir_name}_dataset.pt"]
    
    

    # Process y_data if provided
    if args.y_data:
        # Determine if input is CSV or JSON and convert if necessary
        y_data_file = args.y_data
        if args.y_data.lower().endswith('.csv'):
            # Convert CSV to JSON
            y_data_file = os.path.join(args.data_dir, 'y_data_converted.json')
            y_data_file = convert_csv_to_json(args.y_data, y_data_file)
        elif not args.y_data.lower().endswith('.json'):
            print(f"Error: Unsupported file type. Please provide a CSV or JSON file.")
            sys.exit(1)
        
        if not os.path.exists(y_data_file):
            print(f"Error: Y data file {y_data_file} does not exist.")
            sys.exit(1)
        
        dataset_command.extend(["--data_dict", y_data_file])
    


    # Add dataset construction command to workflow
    workflow_commands.append(dataset_command)

    for command in workflow_commands:
        run_command(command)
    
    print("Dataprep workflow completed successfully!")

if __name__ == "__main__":
    main()