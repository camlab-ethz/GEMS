import re
import json
from collections import Counter

# Assuming 'file_path' is the path to your CSV file
file_path = 'dataset_cleaning_DTI5_c5.csv'

# Pattern to find "test dataset complex " followed by any four characters
pattern = r'test dataset complex (\w{4})'

# Counter to track occurrences of each four-character string
counter = Counter()

# Open and read the file
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Find all occurrences of the pattern in the current line
        matches = re.findall(pattern, line)
        # Update the counter with the found four-character strings
        for match in matches:
            counter[match] += 1

# Convert the counter to a dictionary and write it to a JSON file
with open('test_dataset_counterparts.json', 'w', encoding='utf-8') as json_file:
    json.dump(dict(counter), json_file, ensure_ascii=False, indent=4)

print("Results have been written to results.json")