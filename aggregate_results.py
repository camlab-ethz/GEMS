import argparse
import os
import glob
import csv
import re

def find_last_saved_line(file_path):
    """Find the last line containing 'Saved' in the file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if "Saved" in line:
                return line
    return None

def parse_line(line):
    """Extract data from the line."""
    pattern = r"Epoch (\d+):.*Loss:\s+(\d+.\d+).*Pearson:\s+(\d+.\d+).*R2:\s+(\d+.\d+).*RMSE:\s+(\d+.\d+).*Loss:\s+(\d+.\d+).*Pearson:\s+(\d+.\d+).*R2:\s+(\d+.\d+).*RMSE:\s+(\d+.\d+)"
    match = re.search(pattern, line)
    return match.groups() if match else None

def main():
    parser = argparse.ArgumentParser(description="Parse and write data from text files to a CSV.")
    parser.add_argument("directory", help="Directory path to search for files")
    args = parser.parse_args()

    output_csv = os.path.join(args.directory, 'output.csv')
    columns = ["Epoch", "Train Loss", "Train Pearson", "Train R2", "Train RMSE", "Val Loss", "Val Pearson", "Val R2", "Val RMSE"]

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(columns)

        for i in range(5):
            fold_path = os.path.join(args.directory, f"Fold{i}")
            text_files = glob.glob(os.path.join(fold_path, "train*.out"))

            for file_path in text_files:
                last_saved_line = find_last_saved_line(file_path)
                if last_saved_line:
                    data = parse_line(last_saved_line)
                    if data:
                        csvwriter.writerow(data)

if __name__ == "__main__":
    main()
