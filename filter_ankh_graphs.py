import os

template_path = '/data/grbv/PDBbind/DTI_5_c4b/input_graphs_esm2_t6_8M/training_data'
destination_path = '/data/grbv/PDBbind/DTI_5_c4b/input_graphs_ankh_base/training_data'


template_ids = [filename[0:4] for filename in os.listdir(template_path) if filename.endswith('.pt')]
#print(template_ids)

# Iterate over the destination path and delete a file if it is not in the template_ids list

for filename in os.listdir(destination_path):
    if filename.endswith('.pt') and filename[0:4] not in template_ids:
        os.remove(os.path.join(destination_path, filename))
        print(f"Removed {filename}")