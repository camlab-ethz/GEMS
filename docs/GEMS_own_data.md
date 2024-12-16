# Run GEMS On Your Own Data

Follow these steps to construct a dataset of affinity-labeled interaction graphs from your protein-ligand complexes and run training or inference with GEMS.

* **Prepare your data:**
    * Save PDB and the SDF files of your dataset in a directory.
    * Each protein-ligand pair should share the same unique identifier (_ID_) as filenames to indicate they form a complex. For example, use filenames like _ID_.pdb and _ID_.sdf to represent the same complex.

* **Prepare the labels:**
    * Create a JSON file mapping each ID to its log_kd_ki affinity values, as seen in the example PDBbind_data/PDBbind_data_dict.json.
    * Alternatively, if your data is in CSV format, ensure the file includes columns named _key_ (complex identifier) and _log_kd_ki_ (affinity values). Convert this CSV to a JSON dictionary using the following command:
        ```
        python -m utils.convert_csv_to_json --input_file <path/to/CSV> --output_file <path/to/output/JSON>
        ```

* **Compute Language Model Embeddings:** <br />
To compute ChemBERTa-77M, ANKH-Base and ESM2-T6 embeddings for your data, exectute the following commands:

    * ChemBERTa:     ```python -m dataprep.chemberta_features --data_dir <path/to/data/dir> --model ChemBERTa-77M-MLM``` <br />
    * ANKH:          ```python -m dataprep.ankh_features --data_dir <path/to/data/dir> --ankh_base True``` <br />
    * ESM2:          ```python -m dataprep.esm_features --data_dir <path/to/data/dir> --esm_checkpoint t6``` <br />

    You can also include only a subset of these embeddings or change to ChemBERTa-10M (--model ChemBERTa-10M-MLM), to ANKH-Large (--ankh_base False) or 
    to ESM2-T33 (--esm_checkpoint t33). We recommend running these scripts on a GPU.
  
* **Run the graph construction:** <br />
Construct interaction graphs for all protein-ligand complexes in your data directory, incorporating the desired language model embeddings. For example:
    ```
    python -m dataprep.graph_construction --data_dir <data/dir> --protein_embeddings ankh_base esm2_t6 --ligand_embeddings ChemBERTa_77M
    ```
  
* **Run the dataset construction:** <br />
To create datasets of affinity-labeled interaction graphs, run the command below with the following inputs:
    * path to the directory containing your data (--data_dir)
    * path to save the dataset (--save_path).
    * To include the labels, provide also the path to the JSON file containing the log_kd_ki values.
    * Add the protein embeddings and the ligand embeddings that should be used to featurize the graphs (any combination of the embeddings included in the graph construction is possible, as these embeddings are not yet incorporated into the graph features. This allows you to create datasets with many different combinations of embeddings using the same collection of graph objects).  <br /> <br />
    ```
    python -m dataprep.construct_dataset --data_dir <data/dir> --save_path <output/path/.pt> --data_dict <path/to/dict> --protein_embeddings ankh_base esm2_t6 --ligand_embeddings ChemBERTa_77M
    ```

  
* **Inference/Training:** You can now run inference or training on the generated PyTorch datasets:
    ```
    python inference.py --dataset_path <path/to/dataset>
    ```
    ```
    python train.py --dataset_path <path/to/dataset_file> --run_name <select a run name>
    ```

* **Test:** Test the newly trained model with `test.py`, using the saved stdict and the path to a test dataset as input. If you want to test an ensemble of several models, provide all stdicts in a comma-separated string.
    ```
    python test.py --dataset_path <path/to/downloaded/test/set> --stdicts <path/to/saved/stdict>
    ```
