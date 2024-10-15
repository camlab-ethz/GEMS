#!/bin/bash
#SBATCH --job-name=PYG_GNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --tmp=20G
#SBATCH --gres=gpumem:20G

module load eth_proxy

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="acb1e10c7862354093ceaae70693e1a66403a4ca"
source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate BAP

#---------------------------------
# General Parameters
#---------------------------------

working_dir="/cluster/work/math/dagraber/DTI"
cd $working_dir
data_dir="/cluster/work/math/dagraber/DTI/PDBbind_v2021"
project_name="PDBbind2021"

split="c9"
dataset="L0AE"
data_dict="PDBbind_data/PDBbind_data_dict.json"

# INCLUDE A MASTERNODE
masternode="False"
masternode_connectivity="all"
masternode_edges="undirected"

# EMBEDDINGS TO BE INCLUDED
protein_embeddings="ankh_large" #amino acid embeddings as space-separated strings
ligand_embeddings="ChemBERTa_77M" #ligand embeddings as space-separated strings

# ABLATION STUDY
atom_features="True"
edge_features="True"
delete_ligand="False"
delete_protein="False"

#---------------------------------
# Training
#---------------------------------

# --- MODEL AND TRAINING PARAMETERS ------------
wandb="True"
num_epochs=100
fold_to_train=(0 1 2 3 4)

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

ablation=""
if [ "$delete_protein" = "False" ]; then ablation+="P"
else ablation+="0"
fi

if [ "$delete_ligand" = "False" ]; then ablation+="L"
else ablation+="0"
fi


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

data_split="${working_dir}/PDBbind_data_splits/PDBbind_${split}_data_split.json"
dataset_path="${working_dir}/experiments_v2021/${split}/${dataset}${split}${ablation}"
#dataset_path="${working_dir}/experiments_v2021/${split}/${dataset}${split}${mn}"


# Check if dataset_path already exists
echo "Checking for datasets at $dataset_path"
if [ -d "$dataset_path" ]; then
    echo "The dataset directory already exists"
else
    mkdir -p $dataset_path
    echo "The dataset directory has been created"
fi


train_dataset_path="${dataset_path}/dataset_train.pt"
if [ ! -f "$train_dataset_path" ]; then
    echo "Constructing Training Dataset at $train_dataset_path"
    python dataprep3_construct_dataset.py \
        --data_dir $data_dir \
        --data_dict $data_dict \
        --data_split $data_split \
        --dataset train \
        --save_path $train_dataset_path \
        ${protein_embeddings:+--protein_embeddings $protein_embeddings} \
        ${ligand_embeddings:+--ligand_embeddings $ligand_embeddings} \
        --masternode $masternode --masternode_connectivity $masternode_connectivity --masternode_edges $masternode_edges \
        --atom_features $atom_features --edge_features $edge_features \
        --delete_ligand $delete_ligand --delete_protein $delete_protein
    echo "Construction of Training Dataset complete"
else
    echo "Training dataset already exists at $train_dataset_path. Skipping dataset construction."
fi


casf2013_dataset_path="${dataset_path}/dataset_casf2013.pt"
if [ ! -f "$casf2013_dataset_path" ]; then
    echo "Constructing CASF2013 Dataset at $casf2013_dataset_path"
    python dataprep3_construct_dataset.py \
        --data_dir $data_dir \
        --data_dict $data_dict \
        --data_split $data_split \
        --dataset casf2013 \
        --save_path $casf2013_dataset_path \
        ${protein_embeddings:+--protein_embeddings $protein_embeddings} \
        ${ligand_embeddings:+--ligand_embeddings $ligand_embeddings} \
        --masternode $masternode --masternode_connectivity $masternode_connectivity --masternode_edges $masternode_edges \
        --atom_features $atom_features --edge_features $edge_features \
        --delete_ligand $delete_ligand --delete_protein $delete_protein
    echo "Construction of CASF2013 Dataset complete"
else
    echo "CASF2013 dataset already exists at $casf2013_dataset_path. Skipping dataset construction."
fi


casf2016_dataset_path="${dataset_path}/dataset_casf2016.pt"
if [ ! -f "$casf2016_dataset_path" ]; then
    echo "Constructing CASF2016 Dataset at $casf2016_dataset_path"
    python dataprep3_construct_dataset.py \
        --data_dir $data_dir \
        --data_dict $data_dict \
        --data_split $data_split \
        --dataset casf2016 \
        --save_path $casf2016_dataset_path \
        ${protein_embeddings:+--protein_embeddings $protein_embeddings} \
        ${ligand_embeddings:+--ligand_embeddings $ligand_embeddings} \
        --masternode $masternode --masternode_connectivity $masternode_connectivity --masternode_edges $masternode_edges \
        --atom_features $atom_features --edge_features $edge_features \
        --delete_ligand $delete_ligand --delete_protein $delete_protein
    echo "Construction of CASF2016 Dataset complete"
else
    echo "CASF2016 dataset already exists at $casf2016_dataset_path. Skipping dataset construction."
fi


casf2013_c5_dataset_path="${dataset_path}/dataset_casf2013_c5.pt"
if [ ! -f "$casf2013_c5_dataset_path" ]; then
    echo "Constructing CASF2013 Dataset at $casf2013_c5_dataset_path"
    python dataprep3_construct_dataset.py \
        --data_dir $data_dir \
        --data_dict $data_dict \
        --data_split $data_split \
        --dataset casf2013_c5 \
        --save_path $casf2013_c5_dataset_path \
        ${protein_embeddings:+--protein_embeddings $protein_embeddings} \
        ${ligand_embeddings:+--ligand_embeddings $ligand_embeddings} \
        --masternode $masternode --masternode_connectivity $masternode_connectivity --masternode_edges $masternode_edges \
        --atom_features $atom_features --edge_features $edge_features \
        --delete_ligand $delete_ligand --delete_protein $delete_protein
    echo "Construction of CASF2013 (filtered c5) Dataset complete"
else
    echo "CASF2013 (filtered c5) dataset already exists at $casf2013_c5_dataset_path. Skipping dataset construction."
fi


casf2016_c5_dataset_path="${dataset_path}/dataset_casf2016_c5.pt"
if [ ! -f "$casf2016_c5_dataset_path" ]; then
    echo "Constructing CASF2016 Dataset at $casf2016_c5_dataset_path"
    python dataprep3_construct_dataset.py \
        --data_dir $data_dir \
        --data_dict $data_dict \
        --data_split $data_split \
        --dataset casf2016_c5 \
        --save_path $casf2016_c5_dataset_path \
        ${protein_embeddings:+--protein_embeddings $protein_embeddings} \
        ${ligand_embeddings:+--ligand_embeddings $ligand_embeddings} \
        --masternode $masternode --masternode_connectivity $masternode_connectivity --masternode_edges $masternode_edges \
        --atom_features $atom_features --edge_features $edge_features \
        --delete_ligand $delete_ligand --delete_protein $delete_protein
    echo "Construction of CASF2016 (filtered c5) Dataset complete"
else
    echo "CASF2016 (filtered c5) dataset already exists at $casf2016_c5_dataset_path. Skipping dataset construction."
fi
# ------------------------------------------------


#Function to copy results to working storage - Trap to copy results on exit, it will execute copy_results when the script exits
copy_results() {
    mv "${dataset_path}/slurm-${SLURM_JOB_ID}.out" $run_path
    cp "$0" $run_path
    unset WANDB_API_KEY
    }
trap copy_results EXIT


## --- PARSE ARGUMENTS -----------------------------------------------------------------------------
DROPOUT=0.0
CONV_DROPOUT=0.0
model=""
random_seed=0  # Default seed value

# Parse the arguments
model=$1  # Assign the first argument to MODEL
shift  # Remove the first argument (model)

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
        random_seed )       # Match the random_seed argument
                              shift  # Remove the random_seed argument
                              random_seed=$1  # Assign the next argument to RANDOM_SEED
                              ;;
        * )                   # Match any other argument
                              echo "Invalid parameter detected"
                              exit 1  # Exit with an error code
    esac
    shift  # Remove the current argument and move to the next
done

echo "Model Architecture is $model"
echo "Dropout rate is set to $DROPOUT"
echo "Conv Dropout rate is set to $CONV_DROPOUT"
echo "Random seed is set to $random_seed"
#----------------------------------------------------------------------------------------------------

if [ "$alr_lin" = "True" ] && [ "$alr_mult" = "False" ] && [ "$alr_plateau" = "False" ]; then alr="lin"
elif [ "$alr_lin" = "False" ] && [ "$alr_mult" = "True" ] && [ "$alr_plateau" = "False" ]; then alr="mult"
elif [ "$alr_lin" = "False" ] && [ "$alr_mult" = "False" ] && [ "$alr_plateau" = "True" ]; then alr="plat"
else alr="none"
fi


# START CROSS VALIDATION
# =================================================================================
#run_name="${model}_${dataset}${split}${mn}_${lr#*.}${alr}_d${DROPOUT//./}${CONV_DROPOUT//./}"
run_name="${model}_${dataset}${split}${ablation}_d${DROPOUT//./}${CONV_DROPOUT//./}_${random_seed}"
run_path="${dataset_path}/${model}/${run_name}"

echo
echo "Starting run ${run_name}"
echo

all_stdicts=()
for fold in "${fold_to_train[@]}"; do

    results_path="${run_path}/Fold${fold}"
    echo
    echo "Training Fold ${fold}"

    # Train the model on the fold
    # ----------------------------------------------------------------------------------

    # Only train the model of this fold if the results directory does not already exist
    if [ -d "$results_path" ]; then
        echo "Warning: The results directory ${results_path} already exists. Skipping training for this fold."
    else
        mkdir -p $results_path

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
                    --random_seed $random_seed \
                    --alr_lin $alr_lin --start_factor $start_factor --end_factor $end_factor --total_iters $total_iters \
                    --alr_mult $alr_mult --factor $factor \
                    --alr_plateau $alr_plateau --reduction $reduction --patience $patience --min_lr $min_lr \
                    --early_stopping $early_stopping --early_stop_patience $early_stop_patience --early_stop_min_delta $early_stop_min_delta \
                    --fold_to_train $fold"

        cd $working_dir
        $config_train > $train_stdout 2> $train_stderr
    fi
    # ---------------------------------------------------------------------------------

    


    # Evaluate the trained model on the test sets
    # ---------------------------------------------------------------------------------
    echo
    echo "Evaluating Fold ${fold}"
    stdict=$(find $results_path -name "*best_stdict.pt" -print -quit)
    all_stdicts+=("$stdict")

    echo "Looking for Best Model State Dict in $results_path"
    echo "Best Model State Dict: $stdict"

    config_test="python test.py \
                --model_name $run_name \
                --model_arch $model \
                --save_path $results_path \
                --test_dataset_path $casf2013_dataset_path \
                --train_dataset $train_dataset_path \
                --stdict_paths $stdict"
    $config_test

    config_test="python test.py \
                --model_name $run_name \
                --model_arch $model \
                --save_path $results_path \
                --test_dataset_path $casf2016_dataset_path \
                --train_dataset $train_dataset_path \
                --stdict_paths $stdict"
    $config_test

    config_test="python test.py \
                --model_name $run_name \
                --model_arch $model \
                --save_path $results_path \
                --test_dataset_path $casf2013_c5_dataset_path \
                --train_dataset $train_dataset_path \
                --stdict_paths $stdict"
    $config_test

    config_test="python test.py \
                --model_name $run_name \
                --model_arch $model \
                --save_path $results_path \
                --test_dataset_path $casf2016_c5_dataset_path \
                --train_dataset $train_dataset_path \
                --stdict_paths $stdict"
    $config_test

    # ---------------------------------------------------------------------------------

done


# Aggregate the results of all trained folds into a csv file
echo
echo "Aggregating Results into CSV"
python aggregate_results.py $run_path



# Evaluate the ensemble model of all folds on the test sets
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

old_IFS="$IFS"
IFS=','
all_stdicts_string="${all_stdicts[*]}"
IFS="$old_IFS"

echo "Evaluating Ensemble Model"

config_ensemble="python test.py \
            --model_name $run_name \
            --model_arch $model \
            --save_path $run_path \
            --test_dataset_path $casf2013_dataset_path \
            --train_dataset $train_dataset_path \
            --stdict_paths $all_stdicts_string"
$config_ensemble

config_ensemble="python test.py \
            --model_name $run_name \
            --model_arch $model \
            --save_path $run_path \
            --test_dataset_path $casf2016_dataset_path \
            --train_dataset $train_dataset_path \
            --stdict_paths $all_stdicts_string"
$config_ensemble

config_ensemble="python test.py \
            --model_name $run_name \
            --model_arch $model \
            --save_path $run_path \
            --test_dataset_path $casf2013_c5_dataset_path \
            --train_dataset $train_dataset_path \
            --stdict_paths $all_stdicts_string"
$config_ensemble

config_ensemble="python test.py \
            --model_name $run_name \
            --model_arch $model \
            --save_path $run_path \
            --test_dataset_path $casf2016_c5_dataset_path \
            --train_dataset $train_dataset_path \
            --stdict_paths $all_stdicts_string"
$config_ensemble