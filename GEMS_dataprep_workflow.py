#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from utils.convert_csv_to_json import convert_csv_to_json  # Ensure proper import

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
    parser = argparse.ArgumentParser(description="GEMS Data Preparation Workflow Execution")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory for input and output data')
    parser.add_argument('--y_data', type=str, default=None, 
                        help='Optional path to CSV or JSON file containing y data')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        sys.exit(1)

    # Define the workflow commands
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
    
    # Build the dataset construction command
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
    dataset_command = ["python", "-m", "dataprep.construct_dataset", 
                       "--data_dir", args.data_dir, 
                       "--protein_embeddings", "ankh_base", "esm2_t6", 
                       "--ligand_embeddings", "ChemBERTa_77M", 
                       "--save_path", f"{data_dir_name}_dataset.pt"]

    # Process y_data if provided
    if args.y_data:
        y_data_file = args.y_data
        if args.y_data.lower().endswith('.csv'):
            # Convert CSV to JSON if it's in CSV format
            y_data_file = os.path.join(args.data_dir, 'y_data_converted.json')
            y_data_file = convert_csv_to_json(args.y_data, y_data_file)  # Call the conversion function
        elif not args.y_data.lower().endswith('.json'):
            print(f"Error: Unsupported file type. Please provide a CSV or JSON file as defined in our GitHub documentation.")
            sys.exit(1)

        if not os.path.exists(y_data_file):
            print(f"Error: Y data file {y_data_file} does not exist.")
            sys.exit(1)
        
        # Add the y_data file to the dataset command
        dataset_command.extend(["--data_dict", y_data_file])

    # Append the dataset command to the workflow
    workflow_commands.append(dataset_command)

    # Run all workflow commands
    for command in workflow_commands:
        run_command(command)
    
    print("Dataprep workflow completed successfully!")

if __name__ == "__main__":
    main()
