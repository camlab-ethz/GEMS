# Run GEMS On Your Own Data

Follow the steps below to construct a dataset of affinity-labelled interactions graphs and from your protein-ligand complesxes run trainining/inference. 

* **Prepare your data:** Save PDB and the SDF files of your dataset in a directory. Each protein-ligand pair should share the same unique identifier (_ID_) as filenames to indicate they form a complex. For example, use filenames like _ID_.pdb and _ID_.sdf to represent the same complex.

* **Prepare the labels:** Prepare your labels by writing a CSV file mapping _ID_ to pK values (columns "key" and "log_kd_ki"). 

* **Compute Language Model Embeddings:** To compute ChemBERTa-77M, ANKH-Base and ESM2-T6 embeddings and save them in your data directory, you can run the following commands. 

    ChemBERTa:     ```python -m dataprep.chemberta_features --data_dir <path/to/data/dir> --model ChemBERTa-77M-MLM``` <br />
    ANKH:          ```python -m dataprep.ankh_features --data_dir <path/to/data/dir> --ankh_base True``` <br />
    ESM2:          ```python -m dataprep.esm_features --data_dir <path/to/data/dir> --esm_checkpoint t6``` <br />

    You can also include only a subset of these embeddings or change to ChemBERTa-10M (--model ChemBERTa-10M-MLM), to ANKH-Large (--ankh_base False) or 
    to ESM2-T33 (--esm_checkpoint t33). We recommend running these scripts on a GPU.
  
* **Run the graph construction:** To construct interaction graphs for all protein-ligand complexes in your data directory (incorporating language model embeddings), run the following command with the desired combination of protein and ligand embeddings:

    ```
    python -m dataprep.graph_construction
    --data_dir <data/dir>
    --protein_embeddings ankh_base esm2_t6
    --ligand_embeddings ChemBERTa_77M
    ```
  
* **Run the dataset construction:** You need to provide the path to the directory containing your data (--data_dir) and the path to save the dataset (--save_path). To include the labels, provide also the path to the JSON file containing the log_kd_ki values. Finally, add the protein embeddings and the ligand embeddings that should be used to featurize the graphs (any combination of the embeddings included in the graph construction is possible). This will generate a pytorch dataset of affinity-labelled interactions graphs featurized with the desired language model embeddings.

    ```
    python -m dataprep.construct_dataset
    --data_dir <data/dir> 
    --save_path <save/output/path/.pt>
    --data_dict <path/to/dict/with/log_kd_ki>
    --protein_embeddings ankh_base esm2_t6
    --ligand_embeddings ChemBERTa_77M
    ```

  
* **Inference/Training:** You can now run inference or training on the generated PyTorch datasets:
    ```
    python inference.py --dataset_path <path/to/dataset>
    ```
    ```
    python train.py --dataset_path <path/to/dataset_file> --run_name <select a run name>
    ```
