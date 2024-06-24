#!/bin/bash
#SBATCH --job-name=PYG_GNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --tmp=20G
#SBATCH --gres=gpumem:20G

module load eth_proxy

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="acb1e10c7862354093ceaae70693e1a66403a4ca"
source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate PYG

#---------------------------------
# General Parameters
#---------------------------------

working_dir="/cluster/work/math/dagraber/DTI"
data_dir="/cluster/work/math/dagraber/DTI/PDBbind"
project_name="PDBbind"

#---
split="c0"
filtering="f"
dataset="B0AE"
#---

masternode="False"
masternode_connectivity="all"
masternode_edges="undirected"

protein_embeddings="ankh_base" #embeddings as space-separated strings
ligand_embeddings="" #embeddings as space-separated strings

refined_only="False"
exclude_ic50="False"
exclude_nmr="False"
resolution_threshold=5
precision_strict="False"

atom_features="True"
edge_features="True"
delete_ligand="False"
delete_protein="False"

#---------------------------------
# Training
#---------------------------------

# --- MODEL AND TRAINING PARAMETERS ------------
wandb="True"
num_epochs=2000
fold_to_train=(0 1 2 3 4)
model="GAT0bn"

bs=256
lr=0.001
optimizer='SGD'
loss_func="RMSE"
wd=0.001
# ----------------------------------------------


# --- Early Stopping ---------------------------
early_stopping="True"
early_stop_patience=100
early_stop_min_delta=0.5
# ----------------------------------------------



# --- Adaptive Learning Rate --------------------
alr_lin="False"
start_factor=1
end_factor=0.01
total_iters=$num_epochs

alr_mult="False"
factor=0.9984

alr_plateau="False"
reduction=0.1
patience=10
min_lr=0.0001


#---------------------------------
# Dataset Construction
#---------------------------------

mn="mn"
if [ "$masternode" = "False" ]; then mn+="0"
else
    if [ "$masternode_connectivity" = "all" ]; then mn+="a"
    elif [ "$masternode_connectivity" = "ligand" ]; then mn+="l"
    else mn+="p"
    fi

    if [ "$masternode_edges" = "undirected" ]; then mn+="u"
    elif [ "$masternode_edges" = "in" ]; then mn+="i"
    else mn+="o"
    fi
fi

data_split="PDBbind_data_splits/PDBbind_${split}_data_split.json"
dataset_path="${working_dir}/experiments/${split}/${dataset}${split}${filtering}${mn}"


# Check if dataset_path already exists
if [ -d "$dataset_path" ]; then
    echo "The dataset directory already exists"
else
    mkdir -p $dataset_path
    echo "The dataset directory has been created"
fi


train_dataset_path="${dataset_path}/dataset_train.pt"
if [ ! -f "$train_dataset_path" ]; then
    echo "Constructing Training Dataset"
    python PDBbind_construct_dataset.py \
        --data_dir $data_dir \
        --data_split $data_split \
        --dataset train \
        --save_path $train_dataset_path \
        --refined_only $refined_only --exclude_ic50 $exclude_ic50 --exclude_nmr $exclude_nmr \
        --resolution_threshold $resolution_threshold --precision_strict $precision_strict \
        ${protein_embeddings_str:+--protein_embeddings $protein_embeddings_str} \
        ${ligand_embeddings_str:+--ligand_embeddings $ligand_embeddings_str} \
        --masternode $masternode --masternode_connectivity $masternode_connectivity --masternode_edges $masternode_edges \
        --atom_features $atom_features --edge_features $edge_features \
        --delete_ligand $delete_ligand --delete_protein $delete_protein
    echo "Construction of Training Dataset complete"
else
    echo "Training dataset already exists at $train_dataset_path. Skipping dataset construction."
fi


casf2013_dataset_path="${dataset_path}/dataset_casf2013.pt"
if [ ! -f "$casf2013_dataset_path" ]; then
    echo "Constructing CASF2013 Dataset"
    python PDBbind_construct_dataset.py \
        --data_dir $data_dir \
        --data_split $data_split \
        --dataset casf2013 \
        --save_path $casf2013_dataset_path \
        --refined_only $refined_only --exclude_ic50 $exclude_ic50 --exclude_nmr $exclude_nmr \
        --resolution_threshold $resolution_threshold --precision_strict $precision_strict \
        ${protein_embeddings_str:+--protein_embeddings $protein_embeddings_str} \
        ${ligand_embeddings_str:+--ligand_embeddings $ligand_embeddings_str} \
        --masternode $masternode --masternode_connectivity $masternode_connectivity --masternode_edges $masternode_edges \
        --atom_features $atom_features --edge_features $edge_features \
        --delete_ligand $delete_ligand --delete_protein $delete_protein
    echo "Construction of CASF2013 Dataset complete"
else
    echo "CASF2013 dataset already exists at $casf2013_dataset_path. Skipping dataset construction."
fi


casf2016_dataset_path="${dataset_path}/dataset_casf2016.pt"
if [ ! -f "$casf2016_dataset_path" ]; then
    echo "Constructing CASF2016 Dataset"
    python PDBbind_construct_dataset.py \
        --data_dir $data_dir \
        --data_split $data_split \
        --dataset casf2016 \
        --save_path $casf2016_dataset_path \
        --refined_only $refined_only --exclude_ic50 $exclude_ic50 --exclude_nmr $exclude_nmr \
        --resolution_threshold $resolution_threshold --precision_strict $precision_strict \
        ${protein_embeddings_str:+--protein_embeddings $protein_embeddings_str} \
        ${ligand_embeddings_str:+--ligand_embeddings $ligand_embeddings_str} \
        --masternode $masternode --masternode_connectivity $masternode_connectivity --masternode_edges $masternode_edges \
        --atom_features $atom_features --edge_features $edge_features \
        --delete_ligand $delete_ligand --delete_protein $delete_protein
    echo "Construction of CASF2016 Dataset complete"
else
    echo "CASF2016 dataset already exists at $casf2016_dataset_path. Skipping dataset construction."
fi
# ------------------------------------------------


#Function to copy results to working storage - Trap to copy results on exit, it will execute copy_results when the script exits
copy_results() {
    mv "slurm-${SLURM_JOB_ID}.out" $run_path
    cp "$0" $run_path
    unset WANDB_API_KEY
    }
trap copy_results EXIT


## --- PARSE ARGUEMTNS - DROPOUT -------------------------------------------------------
DROPOUT=0.0
CONV_DROPOUT=0.0

# Parse the arguments
while [ "$1" != "" ]; do  # Continue as long as there are arguments
    case $1 in
        dropout )           # Match the dropout argument
                              shift  # Remove the dropout argument
                              DROPOUT=$1  # Assign the next argument to DROPOUT
                              ;;
        conv_dropout )      # Match the conv_dropout argument
                              shift  # Remove the conv_dropout argument
                              CONV_DROPOUT=$1  # Assign the next argument to CONV_DROPOUT
                              ;;
        * )                   # Match any other argument
                              echo "Invalid parameter detected"
                              exit 1  # Exit with an error code
    esac
    shift  # Remove the current argument and move to the next
done

# Use the parameters in your script
echo "Dropout rate is set to $DROPOUT"
echo "Conv Dropout rate is set to $CONV_DROPOUT"
# -----------------------------------------------------------------------------------------


if [ "$alr_lin" = "True" ] && [ "$alr_mult" = "False" ] && [ "$alr_plateau" = "False" ]; then alr="lin"
elif [ "$alr_lin" = "False" ] && [ "$alr_mult" = "True" ] && [ "$alr_plateau" = "False" ]; then alr="mult"
elif [ "$alr_lin" = "False" ] && [ "$alr_mult" = "False" ] && [ "$alr_plateau" = "True" ]; then alr="plat"
else alr="none"
fi


# START CROSS VALIDATION
# =================================================================================

run_name="${model}_${dataset}${split}${filtering}${mn}_${lr#*.}${alr}_d${DROPOUT//./}"
run_path="${dataset_path}/${run_name}"
all_stdicts=()

for fold in "${fold_to_train[@]}"; do
    echo

    results_path="${run_path}/Fold${fold}"

    # Check if results_path already exists
    if [ -d "$results_path" ]; then
        echo "Error: The results directory ${results_path} already exists. Please change the run_name or delete the existing directory."
        exit 1
    fi
    mkdir -p $results_path



    # Train the model on the fold
    # ----------------------------------------------------------------------------------
    echo "Training Fold ${fold}"

    train_stdout="${results_path}/train-${SLURM_JOB_ID}_f${fold}.out"
    train_stderr="${results_path}/train-${SLURM_JOB_ID}_f${fold}.err"
    config_train="python train.py \
                --dataset_path $train_dataset_path \
                --log_path $results_path \
                --model $model \
                --project_name $project_name \
                --wandb $wandb \
                --run_name $run_name \
                --loss_func $loss_func \
                --optim $optimizer \
                --num_epochs $num_epochs \
                --batch_size $bs \
                --learning_rate $lr \
                --weight_decay $wd \
                --dropout $DROPOUT \
                --conv_dropout $CONV_DROPOUT \
                --alr_lin $alr_lin --start_factor $start_factor --end_factor $end_factor --total_iters $total_iters \
                --alr_mult $alr_mult --factor $factor \
                --alr_plateau $alr_plateau --reduction $reduction --patience $patience --min_lr $min_lr \
                --early_stopping $early_stopping --early_stop_patience $early_stop_patience --early_stop_min_delta $early_stop_min_delta \
                --fold_to_train $fold"

    cd $working_dir
    
    $config_train > $train_stdout 2> $train_stderr

    stdict=$(find $results_path -name "*best_stdict.pt" -print -quit)
    #echo "Best Model State Dict: $stdict"
    all_stdicts+=("$stdict")
    # ---------------------------------------------------------------------------------

    


    # Evaluate the trained model on the test sets
    # ---------------------------------------------------------------------------------
    echo "Evaluating Fold ${fold}"

    config_test="python test.py \
                --model_name $run_name \
                --model_arch $model \
                --save_path $results_path \
                --casf2013_dataset $casf2013_dataset_path \
                --casf2016_dataset $casf2016_dataset_path \
                --train_dataset $train_dataset_path \
                --stdict_paths $stdict"
  

    $config_test
    echo
    # ---------------------------------------------------------------------------------

done


# Aggregate the results of all trained folds into a csv file
echo "Aggregating Results"
python aggregate_results.py $run_path
echo



# Evaluate the ensemble model of all folds on the test sets
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

old_IFS="$IFS"
IFS=','
all_stdicts_string="${all_stdicts[*]}"
IFS="$old_IFS"

config_ensemble="python test.py \
            --model_name $run_name \
            --model_arch $model \
            --save_path $run_path \
            --casf2013_dataset $casf2013_dataset_path \
            --casf2016_dataset $casf2016_dataset_path \
            --train_dataset $train_dataset_path \
            --stdict_paths $all_stdicts_string"

echo "Evaluating Ensemble Model"
$config_ensemble
echo


cp "$0" $run_path