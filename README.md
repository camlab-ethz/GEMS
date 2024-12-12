## This is the GitHub repository for the publication: 

#  GEMS: A Generalizable GNN Framework For Protein-Ligand Binding Affinity Prediction Through Robust Data Filtering and Language Model Integration
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Conda](https://img.shields.io/badge/conda-supported-green.svg)](https://docs.conda.io/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)


David Graber [1,2,3], Peter Stockinger[2], Fabian Meyer [2], Siddhartha Mishra [1]§ Claus Horn [4]§, and Rebecca Buller [2]§

<br />
1 Seminar for Applied Mathematics, Department of Mathematics, and ETH AI Center, ETH Zurich, 8092 Zurich, Switzerland
<br />
2 Competence Center for Biocatalysis, Zurich University of Applied Sciences, 8820 Waedenswil, Switzerland
<br />
3 Institute for Computational Life Sciences, Zurich University of Applied Sciences, 8820 Waedenswil, Switzerland
<br />
4 School of Medicine, Yale University, New Haven, CT 06510, USA
<br />
§ corresponding authors


## Overview 
This repository provides all the necessary resources to use GEMS, a graph-based deep learning model for protein-ligand binding affinity prediction. Here we provide instructions for installing dependencies, and detailed guide for preparing datasets, training the model, and running inference. Below we also explain how to use the PDBbind CleanSplit dataset, a refined training dataset based on PDBbind introduced to eliminate data leakage and improve model generalization. Step-by-step examples are provided to help apply GEMS to your own data or benchmark datasets.


## System Requirements
### Hardware Requirements
* Recommended GPU: NVIDIA RTX3090 or higher with at least 24GB VRAM memory. <br />
* Storage: At least 100GB of storage are needed for preprocessing 20'000 protein-ligand complexes.<br />
* CPU: Part of the code (graph construction) profits from parallelization to several CPUs (about 12h for 20'000 protein-ligand complexes on a single CPU)<br />

We have tested the code using a NVIDIA RTX3090 GPU and do not recommend to run training on CPU only or normal desktop PCs.

### Software Requirements
The package has been tested on the following systems:
Ubuntu 22.04 LTS
Ubuntu 24.04 LTS

**Python Dependencies** <br />
We recomment using miniconda3 to setup a virtual environment with python 3.10. This software has been tested using the following package version:
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
### Via Docker image

All dependencies can be installed using the provided Dockerfile.

Please copy the data on which you want to train, test or predict inside this folder before running the following commands:

```
docker build -t my-gems-container .
docker run --shm-size=8g --gpus all -it my-gems-container
```

### Via conda environment
Alternatively, you can create your conda environment from scratch with the following commands:

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
Optional for training
```
conda install wandb --channel conda-forge
```

## PDBbind CleanSplit and GEMS
PDBbind CleanSplit is a refined training dataset for binding affinity prediction models that is based on PDBbind and has been filtered to reduce redundancy and train-test data leakage into the CASF benchmark datasets. The composition of PDBbind CleanSplit can be found in `PDBbind_data/PDBbind_data_split_cleansplit.json`. 

We provide PyTorch datasets of precomputed interaction graphs for **PDBbind CleanSplit**, for the complete **PDBbind** database (v.2020) and for the **CASF benchmarks** on Zenodo (https://doi.org/10.5281/zenodo.14260171). 

For more information on the available GEMS variants and published datasets, see [GEMS variants and datasets](docs/GEMS_variants_and_datasets.md)



## Run GEMS on example dataset <br />
This repository includes two example datasets of protein-ligand complexes, where each complex comprises a protein stored as a PDB file and a ligand stored as an SDF file. Below are the steps to run inference or training using these provided datasets.

* **Dataset Construction:** Use the `GEMS_dataprep_workflow.py` script to preprocess the data and construct the PyTorch dataset. This script generates interaction graphs enriched with language model embeddings (e.g., esm2_t6, ankh_base, and ChemBERTa-77M). Specify the path to your data directory (containing PDB and SDF files) as an argument. If you wish to include affinity labels for training, provide the path to your labels file (CSV or JSON) as an additional input.
    ```
    python GEMS_dataprep_workflow.py --data_dir example_dataset --y_data PDBbind_data/PDBbind_data_dict.json
    ```

* **Inference:** Run `inference.py` with the newly generated dataset file as input. This file will load the appropriate model and the dataset and create a CSV file containing pK predictions. If the dataset contains labels, it will produce a prediction scatterplot.
    ```
    python inference.py --dataset_path example_dataset_dataset.pt
    ```
    
* **Training:** Run `training.py` with the newly generated dataset file and a chosen run name as inputs. The script will split the data into training and validation datasets, train GEMS on the training dataset, and validate it on the validation set. A new folder named after the run name will be created to save the training outputs.
    ```
    python train.py --dataset_path example_dataset_dataset.pt --run_name example_dataset_train_run
    ```


## Run GEMS on precomputed PDBbind dataset (Zenodo)

We provide PyTorch datasets of precomputed interaction graphs for PDBbind CleanSplit, for the complete PDBbind database (v.2020) and for the CASF benchmarks on Zenodo (https://doi.org/10.5281/zenodo.14260171). Each PyTorch dataset is available in five versions containing different combinations of language model embeddings in the graph features. After downloading the the pytorch datasets (.pt files), you can easily run inference on the datasets.
```
python inference.py --dataset_path <path/to/downloaded/dataset_file>
```

To retrain GEMS on a downloaded pytorch dataset, run the following command with the desired run name:
```
python train.py --dataset_path <path/to/downloaded/dataset_file>  --run_name downloaded_dataset_train_run
```


## Run GEMS on PDBbind (without precomputed datasets) 

If you're interested in creating interaction graph datasets from the PDBbind source data, see our [PDBbind from scratch instructions](docs/GEMS_pdbbind.md).


## Run GEMS on your own data
If you would run GEMS on your own protein-ligand complexes, see our [Run On Your Data Instructions](docs/GEMS_own_data.md)


## Dataset Filtering
This repository includes code of a filtering algorithm that has been used to generate PDBbind CleanSplit. If you are interested in filtering the PDBbind database yourself or filter your own dataset of protein-ligand complexes, see our [Dataset Filtering Instructions](docs/dataset_filtering.md


## License
Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.

## Citation
Please cite the following publication if you found this ressource helpful:

