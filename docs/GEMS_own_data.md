# Run GEMS On Your Own Data

Follow these steps to construct a dataset of affinity-labeled interaction graphs from your protein-ligand complexes and run training or inference with GEMS.
## Prepare your protein-ligand complexes
* **Prepare your data:**
    * Save PDB and the SDF files of your dataset in a directory.
    * Each protein-ligand pair should share the same unique identifier (_ID_) as filenames to indicate they form a complex. For example, use filenames like _ID_.pdb and _ID_.sdf to represent the same complex.
    * If you have several ligands binding to the same protein, an SDF may also contain more than one molecule structure.

* **Prepare the labels (if there are any):**
    * Create a JSON file mapping _IDs_ to _log_kd_ki_ affinity values, as seen in the example `PDBbind_data/PDBbind_data_dict.json`.
    * Alternatively, if your data is in CSV format, ensure the file includes columns named _key_ (ID) and _log_kd_ki_ (affinity pK values). Convert this CSV to a JSON dictionary using the following command:
        ```
        python -m utils.convert_csv_to_json --input_file <path/to/CSV> --output_file <path/to/output/JSON>
        ```

## Graph and Dataset Construction

To create a dataset of interaction graphs ready for inference and training from your data directory, you can use the GEMS_dataprep_workflow.py script. By default, this script incorporates the ESM2-T6, ANKH-Base, and ChemBERTa-77M language model embeddings. To include affinity labels for training, specify the path to your labels file (CSV or JSON) using the --y_data argument:
```
python GEMS_dataprep_workflow.py --data_dir <path/to/data/dir> --y_data <path/to/dict/json>
```

### Custom Language Model Embeddings
If you prefer to use a custom combination of language model embeddings, follow these steps instead:

* **Compute Language Model Embeddings:** <br />
To compute ChemBERTa-77M, ANKH-Base and ESM2-T6 embeddings for your data, exectute the following commands:

    * ChemBERTa:     ```python -m dataprep.chemberta_features --data_dir <path/to/data/dir> --model ChemBERTa-77M-MLM``` <br />
    * ANKH:          ```python -m dataprep.ankh_features --data_dir <path/to/data/dir> --ankh_base True``` <br />
    * ESM2:          ```python -m dataprep.esm_features --data_dir <path/to/data/dir> --esm_checkpoint t6``` <br />

   You can also compute more embeddings, only a subset of these embeddings, or change to ChemBERTa-10M (--model ChemBERTa-10M-MLM), to ANKH-Large (-- 
   ankh_base False) or to ESM2-T33 (--esm_checkpoint t33). We recommend running these scripts on a GPU.
  
* **Graph construction:** <br />
Construct interaction graphs for all protein-ligand complexes in your data directory, incorporating the desired language model embeddings. For example:
    ```
    python -m dataprep.graph_construction --data_dir <data/dir> --protein_embeddings ankh_base esm2_t6 --ligand_embeddings ChemBERTa_77M
    ```
  
* **Dataset construction:** <br />
To create datasets of affinity-labeled interaction graphs, run the command below with the following inputs:
    * path to the directory containing your data (--data_dir)
    * path to save the dataset (--save_path).
    * To include the labels, provide also the path to the JSON file containing the log_kd_ki values (--data_dict)
    * Add the protein embeddings and the ligand embeddings that should be used to featurize the graphs (any combination of the embeddings included in the graph construction is possible, as these embeddings are not yet incorporated into the graph features. This allows you to create datasets with many different combinations of embeddings using the same collection of graph objects).  <br /> <br />
    ```
    python -m dataprep.construct_dataset --data_dir <data/dir> --save_path <output/path/.pt> --data_dict <path/to/dict> --protein_embeddings ankh_base esm2_t6 --ligand_embeddings ChemBERTa_77M
    ```

## Inference and Training
Once your dataset preparation is complete, you can run inference, training, or testing on the newly generated PyTorch datasets using the following commands.

* **Inference:** To predict binding affinities using a pre-trained model:
    ```
    python inference.py --dataset_path <path/to/dataset>
    ```
    
* **Training:** To train GEMS on your dataset, provide the path to the dataset and a unique run name. This script splits the data into a training set (80%) and validation set (20%), trains GEMS on the training set, and evaluates it on the validation set. The model outputs, logs, and checkpoints will be saved in a directory named after your specified run_name. To train with cross-validation, run the command below multiple times, specifying different values for the --fold_to_train argument. For additional training options and parameters, refer to the argparse inputs in the `train.py` script.
    ```
    python train.py --dataset_path <path/to/dataset_file> --run_name <select unique run name>
    ```

* **Test:** Test a trained model using the test.py script. Provide the path to the test dataset and the saved state dictionary (stdict) of your trained model:
    ```
    python test.py --dataset_path <path/to/downloaded/test/set> --stdicts <path/to/saved/stdict>
    ```

    To evaluate an ensemble of multiple models, pass all saved state dictionaries as a comma-separated list:
    ```
    python test.py --dataset_path <path/to/test/dataset> --stdicts <path/to/stdict1>,<path/to/stdict2>,<path/to/stdict3>
    ```
