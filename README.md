## This is the GitHub repository for the publication: 

#  GEMS: A Generalizable GNN Framework For Protein-Ligand Binding Affinity Prediction Through Robust Data Filtering and Language Model Integration
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Conda](https://img.shields.io/badge/conda-supported-green.svg)](https://docs.conda.io/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)


David Graber [1,2,3], Peter Stockinger[2], Fabian Meyer [2], Siddhartha Mishra [1]§ Claus Horn [4]§, and Rebecca Buller [2]§

<sub>1 Seminar for Applied Mathematics, Department of Mathematics, and ETH AI Center, ETH Zurich, 8092 Zurich, Switzerland</sub><br />
<sub>2 Competence Center for Biocatalysis, Zurich University of Applied Sciences, 8820 Waedenswil, Switzerland</sub><br />
<sub>3 Institute for Computational Life Sciences, Zurich University of Applied Sciences, 8820 Waedenswil, Switzerland</sub><br />
<sub>4 School of Medicine, Yale University, New Haven, CT 06510, USA</sub><br />
<sub>§ corresponding authors, shared senior authorship</sub>
<br /> <br /> 
Preprint: https://www.biorxiv.org/content/10.1101/2024.12.09.627482v1 <br />
Dataset: https://doi.org/10.5281/zenodo.14260171 

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
Precomputed datasets for **PDBbind CleanSplit**, the full **PDBbind database** (v.2020), and the **CASF benchmarks** are available on [Zenodo](https://doi.org/10.5281/zenodo.14260171). For details on the provided datasets, refer to [GEMS Variants and Datasets](docs/GEMS_variants_and_datasets.md).

* **Filtering Algorithm:**  <br />
The filtering algorithm that created PDBbind CleanSplit is included in this repository. To run the algorithm, refer to [Filtering Instructions](docs/dataset_filtering.md).  <br />   <br />


## Search Algorithm for Detecting Data Leakage
Some deep-learning-based binding affinity prediction models are outperformed on the CASF benchmarks by a simple search algorithm. This algorithm identifies the most structurally similar training complexes in PDBbind and averages their affinities, exposing data leakage between PDBbind and the CASF2016 benchmark. In contrast, the performance of this algorithm is significantly reduced when tested on PDBbind CleanSplit, a refined dataset specifically designed to eliminate data leakage into the CASF benchmark datasets. To test the performance of the search algorithm, navigate to the `PDBbind_search_algorithm/` directory and execute the following commands:

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

### Run GEMS on example dataset
This repository includes an example dataset of protein-ligand complexes, where each complex comprises a protein (PDB file) and a ligand (SDF file). Follow these steps to run inference or training using the example dataset.

* **Dataset Construction:**  <br />
    Preprocess data and create a PyTorch dataset using `GEMS_dataprep_workflow.py`: This script generates interaction graphs enriched with language model embeddings (e.g., esm2_t6, ankh_base, and ChemBERTa-77M). Specify the path to your data directory (containing PDB and 
    SDF files) as an argument. If you wish to include affinity labels for training, provide the path to your labels file (CSV or JSON) as an additional input.
    ```
    python GEMS_dataprep_workflow.py --data_dir example_dataset --y_data PDBbind_data/PDBbind_data_dict.json
    ```
    
* **Inference:**  <br />
    Generate predictions for the newly generated dataset with `inference.py`. This script will load the appropriate model and the dataset and create a CSV file containing pK predictions. If the dataset contains labels, it will produce a prediction scatterplot.
    ```
    python inference.py --dataset_path example_dataset_dataset.pt
    ```
    
* **Training:**  <br />
    Train the model on the newly generated dataset with `training.py`. This script splits the data into a training set (80%) and validation set (20%), trains GEMS on the training set, and evaluates it on the validation set. Training outputs, including models and logs, are 
    saved in a new folder named after the specified run name:
    ```
    python train.py --dataset_path example_dataset_dataset.pt --run_name example_dataset_train_run
    ```


### Run GEMS on precomputed PDBbind dataset (Zenodo)

Download PyTorch datasets of precomputed interaction graphs from [Zenodo](https://doi.org/10.5281/zenodo.14260171) and run:

* **Inference:**  <br />
    The following command will automatically run inference using the model appropriate for the dataset type:
    ```
    python inference.py --dataset_path <path/to/downloaded/dataset_file>
    ```

* **Training:**  <br />
    For training with cross-validation, run the command below multiple times, specifying different values for the `--fold_to_train` argument. For additional training parameters, refer to the argparse inputs in the `train.py` script:
    ```
    python train.py --dataset_path <path/to/downloaded/train/set>  --run_name downloaded_dataset_train_run
    ```

* **Test:**  <br />
    Test the newly trained model with `test.py`, using the saved stdict and the path to a test dataset as input. If you want to test an ensemble of several models, provide all stdicts in a comma-separated string.
    ```
    python test.py --dataset_path <path/to/downloaded/test/set> --stdicts cleansplit_run/cleansplit_run_f0_best_stdict.pt
    ```
    


### Run GEMS on PDBbind (without precomputed datasets) 
If you're interested in creating interaction graph datasets from the PDBbind source data, see our [PDBbind from scratch instructions](docs/GEMS_pdbbind.md).


### Run GEMS on your own data
For running GEMS on your own protein-ligand complexes, refer to [Run on Your Data](docs/GEMS_own_data.md).


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
