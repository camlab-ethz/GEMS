#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os

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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Machine Learning Workflow Execution")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory for input and output data')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        sys.exit(1)
    
    # Workflow commands with dynamic data directory
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
         "--ligand_embeddings", "ChemBERTa_77M"],
        
        # Dataset Construction
        ["python", "-m", "dataprep.construct_dataset", 
         "--data_dir", args.data_dir, 
         "--protein_embeddings", "ankh_base", "esm2_t6", 
         "--ligand_embeddings", "ChemBERTa_77M", 
         "--save_path", f"{args.data_dir}/dataset.pt"],
        
        # Inference
        ["python", "inference.py", 
         "--model_arch", "GATE18d", 
         "--dataset_path", f"{args.data_dir}/dataset.pt"]
    ]
    
    # Execute each command in sequence
    for command in workflow_commands:
        run_command(command)
    
    print("Workflow completed successfully!")

if __name__ == "__main__":
    main()