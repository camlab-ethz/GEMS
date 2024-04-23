import re
import json
from collections import Counter

# Assuming 'file_path' is the path to your CSV file
file_path = 'dataset_cleaning_DTI5_c4.csv'

# Pattern to find "test dataset complex " followed by any four characters
pattern_test = r'test dataset complex (\w{4})'
pattern_train = r'Removed (\w{4})'

# Counter to track occurrences of each four-character string
conflicts = {}

# Open and read the file
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Find all occurrences of the pattern in the current line
        matches_test = re.findall(pattern_test, line)
        matches_train = re.findall(pattern_train, line)



        # Update the counter with the found four-character strings
        for testp, trainp in zip(matches_test, matches_train):

            if testp not in conflicts.keys():
                conflicts[testp] = [trainp]
            else:
                conflicts[testp].append(trainp)

# Convert the counter to a dictionary and write it to a JSON file
with open('c4_test_dataset_similarities.json', 'w', encoding='utf-8') as json_file:
    json.dump(conflicts, json_file, ensure_ascii=False, indent=4)

print("Results have been written to results.json")