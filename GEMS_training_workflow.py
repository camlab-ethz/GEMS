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
    parser = argparse.ArgumentParser(description="Machine Learning Workflow Execution")
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the dataset.pt file to be used for inference') 
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset {args.dataset_path} does not exist.")
        sys.exit(1)
    

    workflow_commands = [
        
        # Inference
        ["python", "train.py", 
         "--model", "GATE18d", 
         "--run_name", "example_run", 
         "--log_path", "example_run/test_0", 
         "--dataset_path", args.dataset_path]
    ]
    
    for command in workflow_commands:
        run_command(command)
    
    print("Training Workflow completed successfully!")

if __name__ == "__main__":
    main()