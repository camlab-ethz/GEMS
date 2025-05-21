## PDBbind CleanSplit and GEMS
PDBbind CleanSplit is a refined training dataset for binding affinity prediction models that is based on PDBbind and has been filtered to reduce redundancy and train-test data leakage into the CASF benchmark datasets. The composition of PDBbind CleanSplit can be found in `PDBbind_data/PDBbind_data_split_cleansplit.json`. 

We provide PyTorch datasets of precomputed interaction graphs for **PDBbind CleanSplit**, for the complete **PDBbind** database (v.2020) and for the **CASF benchmarks** on Zenodo (https://doi.org/10.5281/zenodo.14260171). Each PyTorch dataset is available in five versions containing different combinations of language model embeddings in the graph features.

### Preprocessed datasets
* `pytorch_datasets_00AEPL` -  ChemBERTa-77M included
* `pytorch_datasets_B0AEPL` -  ChemBERTa-77M and ankh_base included
* `pytorch_datasets_06AEPL` -  ChemBERTa-77M and ESM2-T6 included
* `pytorch_datasets_B6AEPL` -  ChemBERTa-77M, ankh_base and ESM2-T6 included
* `pytorch_datasets_B6AEPL` -  **Ablation:** ChemBERTa-77M, ankh_base and ESM2-T6 included, protein nodes deleted from graph

In addition, we provide GEMS models that have been trained on each of these datasets: 

### Pretrained models
* `model/GEMS18e_00AEPL_d0100` - No embedding included (trained with GEMS18e architecture, neglects ChemBERTa)
* `model/GEMS18d_00AEPL_d0100` - ChemBERTa-77M included
* `model/GEMS18d_B0AEPL_d0600` - ChemBERTa-77M and ankh_base included
* `model/GEMS18d_06AEPL_d0500` - ChemBERTa-77M and ESM2-T6 included
* `model/GEMS18d_B6AEPL_d0500` - ChemBERTa-77M, ankh_base and ESM2-T6 included
* `model/GEMS18d_B6AE0L_d0100` - **Ablation:** ChemBERTa-77M, ankh_base and ESM2-T6 included, protein nodes deleted from graph

For each model, we provide five stdicts corresponding to the models originating from 5-fold cross-validation. Depending on the language model embeddings incorporated, these model showed different performance on benchmark datasets:

![Description](GEMS_stdicts.png)
