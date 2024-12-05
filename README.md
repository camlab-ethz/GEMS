## This is the GitHub Repository for the publication: 

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


## Background
The field of computational drug design requires accurate scoring functions to predict binding affinities for protein-ligand interactions. However, train-test data leakage between the PDBbind database and the CASF benchmark datasets has significantly inflated the performance metrics of currently available deep learning based binding affinity prediction models, leading to overestimation of their generalization capabilities. We address this issue by proposing PDBbind CleanSplit, a training dataset curated by a novel structure-based filtering algorithm that eliminates train-test data leakage as well as redundancies within the training set. Retraining the current best-performing model on CleanSplit caused its benchmark performance to drop to uncompetitive levels, indicating that the performance of existing models is largely driven by data leakage. In contrast, our graph neural network model, GEMS, maintains high benchmark performance when trained on CleanSplit. Leveraging a sparse graph modeling of protein-ligand interactions and transfer learning from language models, GEMS is able to generalize to strictly independent test datasets.

## Overview

In this repository we provide instructions to use the GEMS model for protein-ligand binding affinity prediction. Python scripts are provided for direct execution of dataset construction, training and inference workflows on your own data.

* **Prepare your data:** <br />Ensure that all complexes are stored in the same directory, with proteins saved as PDB files and their corresponding ligands saved as SDF files. Each protein-ligand pair should share the same unique identifier (_ID_) as filenames to indicate they form a complex. For example, use filenames like _ID_.pdb and _ID_.sdf to represent the same complex. If you have affinity labels for your complexes, provide them as a CSV file _(Describe How)_  <br /> <br />
* **Dataset construction:** <br /> Run GEMS_dataprep_workflow.py with the path to your data directory (containing all pairs of PDBs and SDFs) as argument. If you want to add labels (for training), add the path to your labels CSV or JSON file as another input (optional). This creates a pytorch dataset of interaction graphs featurized with language model embeddings (in this case esm2_t6, ankh_base and ChemBERTa-77M). You can now run inference or training on this dataset. <br />
``` python GEMS_dataprep_workflow.py --data_dir <path/to/your/data/dir> --y_data <path/to/labels/file>  ```  <br /> <br />
* **Inference:** <br /> Run GEMS_inference_workflow.py with the path to the generated dataset as input <br />
``` python GEMS_inference_workflow.py --dataset_path <path/to/dataset/pt>```  <br /> <br />
* **Training:** <br /> Run GEMS_training_workflow.py with the path to the generated dataset as input <br />
``` python GEMS_inference_workflow.py --dataset_path <path/to/dataset/pt>```  <br /> <br />

Please note that PDBBind dataset needs to be licensed, which is free for academic users (http://www.pdbbind.org.cn/). 
the code to generate "CleanSplit" dataset from PDBBind, as well as 
 However, we recommend to consider parallel execution of the data preparation scripts if sufficient computing power is available (e.g. on HPC infrastructures for which the user needs to generate own slurm scripts).


## System Requirements
### Hardware Requirements
* Recommended GPU: NVIDIA RTX3090 or higher with at least 24GB VRAM memory. <br />
* Storage: At least 100GB of storage are needed for preprocessing 20'000 protein-ligand complexes.<br />
* CPU: Part of the code (graph construction) profits from parallelization to several CPUs (about 12h for 20'000 protein-ligand complexes on a single CPU)<br />
<br />
We have tested the code using a NVIDIA RTX3090TI GPU<br />

We do not recommend to run the code on CPU only systems or normal desktop PCs.

## Software Requirements
### OS Requirements
The package has been tested on the following systems:
Ubuntu 22.04 LTS
Ubuntu 24.04 LTS

### Python Dependencies
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
sudo docker build -t my-gems-container .

sudo docker run --gpus all -it my-gems-container
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
**Optional for training**
```
conda install wandb --channel conda-forge
```
**Test of installation**<br />

To test the installation we have added a folder with synthetic data. Please execute the following command:

A) example_inference

```
python GEMS_dataprep_workflow.py --data_dir example_dataset 
python GEMS_inference_workflow.py --data_dir example_dataset 
```
B) example_training
For training, path to y_data also needs to be provided for dataset prepation. It can either be provided as csv or as json file. Please note that columns in y_data csv should be: 'key', 'log_kd_ki'
```
python GEMS_dataprep_workflow.py --data_dir example_dataset/ --y_data example_dataset/example_training_data.csv
python GEMS_training_workflow.py --data_dir example_dataset
```


## How to use
**CleanSplit**<br />

Describe here how to apply CleanSplit on PDBBind dataset or own datasets

<br />
**Inference**<br />
To run inference on a set of protein pdbs and ligands, run the following command:<br />
```
python inference.py folder_name<br />
```

<br />
**Training**<br />
To retrain the model on your own data, please execute the following steps:<br />
X
Y
Z
<br />

## Citation
Please cite the following publication if you found this ressource helpful:
add DOI here

## Additional Data Availability
For fast reproduction of our results, we provide PyTorch datasets of precomputed interaction graphs for the entire PDBbind database on Zenodo (https://doi.org/10.5281/zenodo.14260171). To enable quick establishment of leakage-free evaluation setups with PDBbind, we also provide pairwise similarity matrices for the entire PDBbind dataset on Zenodo.
