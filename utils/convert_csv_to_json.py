#!/usr/bin/env python3

import sys
import csv
import json
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Convert a CSV file to JSON.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", required=True, help="Path to the output JSON file")
    args = parser.parse_args()

    # Call the conversion function
    convert_csv_to_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
