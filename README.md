## This is the GitHub repository for the publication: 

#  GEMS - Enhancing Generalizable Binding Affinity Prediction by Removing Data Leakage and Integrating Language Model Embeddings into Graph Neural Networks
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Conda](https://img.shields.io/badge/conda-supported-green.svg)](https://docs.conda.io/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)


David Graber [1,2,3], Peter Stockinger[2], Fabian Meyer [2], Claus Horn [4]§, Siddhartha Mishra [1]§ and Rebecca Buller [2]§

<sub>1 Seminar for Applied Mathematics, Department of Mathematics, and ETH AI Center, ETH Zurich, 8092 Zurich, Switzerland</sub><br />
<sub>2 Competence Center for Biocatalysis, Zurich University of Applied Sciences, 8820 Waedenswil, Switzerland</sub><br />
<sub>3 Institute for Computational Life Sciences, Zurich University of Applied Sciences, 8820 Waedenswil, Switzerland</sub><br />
<sub>4 School of Medicine, Yale University, New Haven, CT 06510, USA</sub><br />
<sub>§ corresponding authors, shared senior authorship</sub>
<br /> <br /> 
Preprint: https://www.biorxiv.org/content/10.1101/2024.12.09.627482v1 <br />
Dataset: https://doi.org/10.5281/zenodo.15482796

## Overview 
This repository provides all resources required to use **GEMS**, a graph-based deep learning model designed for protein-ligand binding affinity prediction. It includes instructions for installing dependencies, preparing datasets, training the model, and running inference. The repository also features **PDBbind CleanSplit**, a refined training dataset based on PDBbind that minimizes data leakage and enhances model generalization. Detailed examples demonstrate how to apply GEMS to your data.


## Hardware Requirements
* **Training and Inference**
	- **GPU:** NVIDIA RTX3090 or higher, with at least 24GB VRAM. We have tested the code using a NVIDIA RTX3090 GPU and do not recommend to run training on CPU only or normal desktop PCs.
	- **Storage:** About 5GB of storage are needed for storing a fully-featurized training dataset of 20'000 interaction graphs.

* **Graph and Dataset Construction** (not needed if precomputed datasets from Zenodo are used)
	- **CPU:** Multi-core processors are recommended; graph construction takes ~12 hours for 20,000 complexes on a single CPU
	- **Storage:** At least 100GB of storage are needed for preprocessing 20'000 protein-ligand complexes.

## Software Requirements
The code has been tested on the following systems:
- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS

**Python Dependencies** <br />
We recommend using `miniconda3` to set up a Python 3.10 virtual environment. This software has been tested with:
```
python=3.10.8
numpy=1.26.4
rdkit=2024.03.3
transformers=4.33.3
ankh=1.10.0
biopython=1.83
pytorch=2.0.1
pytorch-cuda=11.7
pyg=2.5.2
```

## Installation Guide
### Using Docker
You can install all dependencies using the provided Dockerfile. Ensure your data to train, test or predict is copied into this directory before executing the following commands:
```
docker build -t my-gems-container .
docker run --shm-size=8g --gpus all -it my-gems-container
```

### Using conda environment
Alternatively, create a Conda environment from scratch with the following commands:
```
conda create --name GEMS python=3.10
conda activate GEMS
conda install -c conda-forge numpy rdkit
conda install -c huggingface transformers (ensure a version that supports ESM2)
pip install ankh
conda install biopython
conda install pytorch=2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=*=*cu117 -c pyg
```
Optional for tracking:
```
conda install wandb --channel conda-forge
```

## Dataset Filtering and PDBbind CleanSplit

PDBbind CleanSplit is a refined training dataset for binding affinity prediction models that is based on PDBbind and has been filtered to reduce redundancy and train-test data leakage into the CASF benchmark datasets. The dataset split is available in `PDBbind_data/PDBbind_data_split_cleansplit.json`. 

* **Precomputed Datasets:**  <br />
Precomputed datasets for **PDBbind CleanSplit**, the full **PDBbind database** (v.2020), and the **CASF benchmarks** are available on [Zenodo](https://doi.org/10.5281/zenodo.15482796). For details on the available datasets, refer to [GEMS Variants and Datasets](docs/GEMS_variants_and_datasets.md).

* **Filtering Algorithm:**  <br />
The filtering algorithm that created PDBbind CleanSplit is included in this repository. To run the algorithm, refer to [Filtering Instructions](docs/dataset_filtering.md).

* **Pairwise Similarity Matrices for PDBbind:** <br />
The pairwise Tanimoto similarities, TM-scores and pocket-aligned ligand RMSD values for all PDBbind complexes can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.15482796). See [Filtering Instructions](docs/dataset_filtering.md). <br /> <br />


## Search Algorithm for Detecting Data Leakage
Some deep-learning-based binding affinity prediction models are outperformed on the CASF benchmarks by a simple search algorithm. This algorithm identifies the most structurally similar training complexes in PDBbind and averages their affinities. The performance of this algorithm is significantly reduced when tested on PDBbind CleanSplit, a refined dataset specifically designed to eliminate data leakage into the CASF benchmark datasets. <br /> <br />
To test the performance of the search algorithm, first download the precomputed similarity matrices from [Zenodo](https://doi.org/10.5281/zenodo.15482796) and save them at the following location:

- `pairwise_similarity_matrices/pairwise_similarity_complexes.json`
- `pairwise_similarity_matrices/pairwise_similarity_matrix_tanimoto.npy`
- `pairwise_similarity_matrices/pairwise_similarity_matrix_tm.npy`
- `pairwise_similarity_matrices/pairwise_similarity_matrix_rmsd.npy`


Then, navigate to the `PDBbind_search_algorithm/` directory and execute the following commands:

* **On original PDBbind:**
    ```
    python search_algorithm_compl.py --data_split ../PDBbind_data/PDBbind_data_split_pdbbind.json --test_dataset casf2016
    ```
* **On PDBbind CleanSplit**
    ```
    python search_algorithm_compl.py --data_split ../PDBbind_data/PDBbind_data_split_cleansplit.json --test_dataset casf2016
    ```


## GEMS
GEMS (GNN for Efficient Molecular Scoring) is a graph neural network designed for accurate structure-based protein-ligand binding affinity prediction. Trained on PDBbind CleanSplit, a refined dataset free of train-test leakage and redundancy, GEMS leverages transfer learning from protein language models to achieve robust generalization to independent test datasets. Several trained GEMS models are included in this repository. For details on the GEMS variants, see [GEMS Variants and Datasets](docs/GEMS_variants_and_datasets.md).



### Run GEMS on your own data
To predict binding affinities for your own protein-ligand complexes using GEMS, follow the steps below, which demonstrate the process using an example dataset. For more detailed instructions on using and retraining GEMS with your data, refer to [GEMS your data](docs/GEMS_own_data.md).

* **Prepare your data:** 
    * Save PDB and the SDF files of your dataset in a directory (as in `example_dataset/`).
    * Each protein-ligand pair should share the same unique identifier (_ID_) as filenames to indicate they form a complex. For example, use filenames like _ID_.pdb and _ID_.sdf to represent the same complex.
    * If you have several ligands binding to the same protein, an SDF may also contain more than one molecule structure.

* **Dataset Construction:**  <br />
Preprocess data and create a PyTorch dataset using `GEMS_dataprep_workflow.py`: This script generates a dataset of interaction graphs enriched with language model embeddings (e.g., esm2_t6, ankh_base, and ChemBERTa-77M) from an input diretory.
    ```
    python GEMS_dataprep_workflow.py --data_dir example_dataset
    ```
    
* **Inference:**  <br />
To generate binding affinity predictions for the newly created dataset, use the `inference.py` script. The script loads the appropriate GEMS model, processes the dataset, and outputs predictions as a CSV file containing pK values:
    ```
    python inference.py --dataset_path example_dataset_dataset.pt
    ```


### Run GEMS on PDBbind 
This section explains how to run inference or training of GEMS on the PDBbind database using our precomputed datasets of interaction graphs on [Zenodo](https://doi.org/10.5281/zenodo.15482796). These include **PDBbind CleanSplit**, the complete **PDBbind database** (v.2020) and the **CASF benchmarks**. For more details on available datasets and variants, refer to [GEMS Variants and Datasets](docs/GEMS_variants_and_datasets.md). 

**Note:** If you prefer to start with the PDBbind source data and construct the graphs yourself (e.g., using other language model embeddings), follow the instructions in [PDBbind from scratch](docs/GEMS_pdbbind.md).

* **Download datasets:**  <br />
Download PyTorch datasets from [Zenodo](https://doi.org/10.5281/zenodo.15482796)

* **Inference:**  <br />
    To generate affinity predictions, use the following command. The script will load the appropriate model for the dataset type:
    ```
    python inference.py --dataset_path <path/to/downloaded/dataset/file>
    ```

* **Training:**  <br />
    To train GEMS on the downloaded dataset, execute the command below. This splits the data into a training set (80%) and validation set (20%), trains GEMS on the training set, and evaluates it on the validation set. To train with cross-validation, run the command below multiple times, specifying different values for the --fold_to_train argument. For additional training options and parameters, refer to the argparse inputs in the `train.py` script.
    ```
    python train.py --dataset_path <path/to/downloaded/train/set>  --run_name <select unique run name>
    ```

* **Test:**  <br />
    Test the newly trained model with `test.py`, using the saved stdict and the path to a test dataset as input. If you want to test an ensemble of several models, provide all stdicts in a comma-separated string.
    ```
    python test.py --dataset_path <path/to/downloaded/test/set> --stdicts <path/to/stdict>
    ```
    ```
    python test.py --dataset_path <path/to/test/dataset> --stdicts <path/to/stdict1>,<path/to/stdict2>,<path/to/stdict3>
    ```

## License
This project is licensed under the MIT License. It is freely available for academic and commercial use.

## Citation
If you find this resource helpful, please cite the following publication:

```bibtex
@article {Graber2024.12.09.627482,
	author = {Graber, David and Stockinger, Peter and Meyer, Fabian and Mishra, Siddhartha and Horn, Claus and Buller, Rebecca M. U.},
	title = {GEMS: A Generalizable GNN Framework For Protein-Ligand Binding Affinity Prediction Through Robust Data Filtering and Language Model Integration},
	elocation-id = {2024.12.09.627482},
	year = {2024},
	doi = {10.1101/2024.12.09.627482},
}
```
