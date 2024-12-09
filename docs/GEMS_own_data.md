**Prepare your data:** Ensure that all complexes are stored in the same directory, with proteins saved as PDB files and their corresponding ligands saved as SDF files. Each protein-ligand pair should share the same unique identifier (_ID_) as filenames to indicate they form a complex. For example, use filenames like _ID_.pdb and _ID_.sdf to represent the same complex. 
**Prepare your labels:** If you have affinity labels for your complexes, save them as CSV with two columns. Column 1 header should be "key" and column 2 header should be "log_kd_ki". You can also create a dictionary mapping _ID_ to pK values and save it as a json file following the structure of `PDBbind_data/PDBbind_data_dict.json` 
**Run the data preparation** using  
    ```
    python GEMS_dataprep_workflow.py --data_dir example_dataset_2 --y_data PDBbind_data/PDBbind_data_dict.json
    ```