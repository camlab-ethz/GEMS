## This is the GitHub Repository for the publication: 

#  GEMS: GNN Framework For Efficient Protein-Ligand Binding Affinity Prediction Through Robust Data Filtering and Language Model Integration

David Graber [1,2,3], Peter Stockinger[2], Fabian Meyer [2], Siddhartha Mishra [1]§ Claus Horn [3]§, and Rebecca Buller [2]§

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

In this repository we provide the code to generate "CleanSplit" dataset from PDBBind, as well as workflows for training and inference of the GEMS model.
Please note that PDBBind dataset needs to be licensed, which is free for academic users (http://www.pdbbind.org.cn/). 

Two python scripts are provided for direct execution of full training and inference workflows. However, we recommend to consider parallel execution of the data preparation scripts if sufficient computing power is available (e.g. on HPC infrastructures for which the user needs to generate own slurm scripts).

We provide multiple models.... PLS ADD MORE CONTEXT DAVID

## System Requirements
### Hardware Requirements
We recommend to utilize a server with GPU support (at least XGB VRAM), sufficient memory (at least XGB RAM. At least 100GB of storage are needed, preferably SSD.
Part of the code (PLS ADD MORE CONTEXT DAVID) profits from strong CPU (e.g. XXX).

We have tested the code using the following GPU architecture: NVIDIA RTX3090TI

## Software Requirements
### OS Requirements
The package has been tested on the following systems:
Ubuntu 22.04 LTS
PLS ADD MORE CONTEXT DAVID

### Python Dependencies
We recomment using miniconda3 to setup a virtual environment with python 3.10.

## Installation Guide

**Setup of environment for data preparation and inference**<br />
<br />
**Initialize the environment with python 3.10**<br />
conda create --name GEMS python=3.10<br />
conda activate GEMS<br />
<br />
**Install packages:**<br />
conda install -c conda-forge numpy rdkit <br />
conda install -c huggingface transformers (ensure a version that supports ESM2)<br />
pip install ankh <br />
conda install biopython <br />
conda install pytorch=2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia <br />
conda install pyg=*=*cu117 -c pyg <br />

**Optional for training**<br />
conda install wandb --channel conda-forge<br />

**Installation from Dockerfile**<br />
Alternatively, we provide a Dockerfile/Dockerimage ... PLS ADD MORE CONTEXT PETER

Please copy the data on which you want to train, test or predict inside this folder before running the following commands:

sudo docker build -t my-gems-container .

sudo docker run --gpus all -it my-gems-container


**Test of installation**<br />

To test the installation we have added a folder with synthetic data. Please execute the following command:

A) example_inference

command: python GEMS_prediction_workflow.py --data_dir example_inference 

B) example_training

command: python GEMS_training_workflow.py --data_dir example_training



## How to use
**CleanSplit**<br />

PLS ADD MORE CONTEXT DAVID
Describe here how to apply CleanSplit on PDBBind dataset or own datasets

<br />
**Inference**<br />
To run inference on a set of protein pdbs and ligands, run the following command:<br />
python inference.py folder_name<br />



<br />
**Training**<br />
To retrain the model on your own data, please execute the following steps:<br />
X
Y
Z
<br />
PLS ADD MORE CONTEXT DAVID

## Citation
Please cite the following publication if you found this ressource helpful:
add DOI here

## Additional Data Availability
Please cite the following publication if you found this ressource helpful:
add DOI here
