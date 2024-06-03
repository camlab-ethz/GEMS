#!/bin/bash
#SBATCH --job-name=PYG_GNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --tmp=20G
#SBATCH --gres=gpumem:20G

module load eth_proxy

export WANDB__SERVICE_WAIT=300
export WANDB_API_KEY="acb1e10c7862354093ceaae70693e1a66403a4ca"
source /cluster/project/math/dagraber/miniconda3/etc/profile.d/conda.sh
conda activate PYG


# Define variables for the parameters
project_name="DTI5"
filtering="c4"
wandb="True"

#############################

experiment="DTI5f_c4"
run_name="00_testrun_all_in_one"

conv_dropout=0
dropout=0

num_epochs=10
fold_to_train=(0 1 2 3 4)

masternode="all"
model="GAT4mnbn"

# Dataset Composition
emb="False"
embedding_descriptor="ankh_base"
ef="True"
af="True"
refined_only="False"
exclude_ic50="False"
exclude_nmr="False"
resolution_threshold=3.0
precision_strict="False"

# If either protein nodes or ligand nodes should be omitted (not both)
# Should only be used if masternode = "all"
delete_protein="False"
delete_ligand="False"


# Training Hyperparameters
bs=256
lr=0.001
optimizer='SGD'
loss_func="RMSE"
wd=0.001

# Early Stopping
early_stopping="True"
early_stop_patience=100
early_stop_min_delta=0.5

#############################

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

#############################

working_dir="/cluster/work/math/dagraber/DTI"
test_data_dir="/cluster/work/math/dagraber/DTI/test_data/"

# Path to data - Unpack the data into temporary directory in local scratch
archive_file="${project_name}${filtering}_train_data_${embedding_descriptor}.tar.gz"
DATA_ARCHIVE="${working_dir}/training_data/${project_name}${filtering}/${archive_file}"
cp $DATA_ARCHIVE $TMPDIR
echo "Training Data ${archive_file} moved to ${TMPDIR}"
cd $TMPDIR
tar -xzf $archive_file
echo "Training Data unpacked"

# Path to the unpacked data
data_dir="${TMPDIR}/training_data/"


# Create a folder in the working directory for the run
experiment_path="${working_dir}/experiments/${experiment}/${run_name}/"

#Function to copy results to working storage - Trap to copy results on exit, it will execute copy_results when the script exits
copy_results() {
    mv "slurm-${SLURM_JOB_ID}.out" $experiment_path
    cp "$0" $experiment_path
    unset WANDB_API_KEY
    }
trap copy_results EXIT


all_stdicts=()


# START CROSS VALIDATION
# =================================================================================

for fold in "${fold_to_train[@]}"; do
    echo

    results_path="${experiment_path}Fold${fold}/"

    # Check if results_path already exists
    if [ -d "$results_path" ]; then
        echo "Error: The results directory ${results_path} already exists. Please change the run_name or delete the existing directory."
        exit 1
    fi
    mkdir -p $results_path



    # Train the model on the fold
    # ----------------------------------------------------------------------------------
    echo "Training Fold ${fold}"

    train_stdout="${results_path}train-${SLURM_JOB_ID}_f${fold}.out"
    train_stderr="${results_path}train-${SLURM_JOB_ID}_f${fold}.err"
    config_train="python train.py \
                --data_dir $data_dir --log_path $results_path --model $model --project_name $project_name --wandb $wandb --run_name $run_name \
                --embedding $emb --edge_features $ef --atom_features $af --masternode $masternode --refined_only $refined_only --exclude_ic50 $exclude_ic50 \
                --resolution_threshold $resolution_threshold --precision_strict $precision_strict --exclude_nmr $exclude_nmr --loss_func $loss_func --optim $optimizer \
                --num_epochs $num_epochs --batch_size $bs --learning_rate $lr --weight_decay $wd --dropout $dropout --conv_dropout $conv_dropout \
                --alr_lin $alr_lin --start_factor $start_factor --end_factor $end_factor --total_iters $total_iters \
                --alr_mult $alr_mult --factor $factor --delete_ligand $delete_ligand --delete_protein $delete_protein \
                --early_stopping $early_stopping --early_stop_patience $early_stop_patience --early_stop_min_delta $early_stop_min_delta \
                --alr_plateau $alr_plateau --reduction $reduction --patience $patience --min_lr $min_lr \
                --fold_to_train $fold"

    cd $working_dir
    
    $config_train > $train_stdout 2> $train_stderr

    stdict=$(find $results_path -name "*best_stdict.pt" -print -quit)
    all_stdicts+=("$stdict")
    # ---------------------------------------------------------------------------------

    


    # Evaluate the trained model on the test sets
    # ---------------------------------------------------------------------------------
    echo "Evaluating Fold ${fold}"

    config_test="python test.py \
                --test_data_dir $test_data_dir \
                --train_data_dir $data_dir \
                --model_name $run_name \
                --model_arch $model \
                --save_dir $results_path \
                --stdict_paths $stdict \
                --filtering $filtering \
                --embedding_descriptor $embedding_descriptor \
                --embedding $emb \
                --edge_features $ef \
                --atom_features $af \
                --masternode $masternode \
                --refined_only $refined_only \
                --exclude_ic50 $exclude_ic50 \
                --resolution_threshold $resolution_threshold \
                --precision_strict $precision_strict \
                --exclude_nmr $exclude_nmr \
                --delete_ligand $delete_ligand \
                --delete_protein $delete_protein"

    $config_test
    echo
    # ---------------------------------------------------------------------------------

done

# Aggregate the results of all trained folds into a csv file
echo "Aggregating Results"
python aggregate_results.py $experiment_path
echo



# Evaluate the ensemble model of all folds on the test sets
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

old_IFS="$IFS"
IFS=','
all_stdicts_string="${all_stdicts[*]}"
#all_stdicts_string="${all_stdicts[@]}"
IFS="$old_IFS"

config_ensemble="python test.py \
            --test_data_dir $test_data_dir \
            --train_data_dir $data_dir \
            --model_name $run_name \
            --model_arch $model \
            --save_dir $experiment_path \
            --stdict_paths $all_stdicts_string \
            --filtering $filtering \
            --embedding_descriptor $embedding_descriptor \
            --embedding $emb \
            --edge_features $ef \
            --atom_features $af \
            --masternode $masternode \
            --refined_only $refined_only \
            --exclude_ic50 $exclude_ic50 \
            --resolution_threshold $resolution_threshold \
            --precision_strict $precision_strict \
            --exclude_nmr $exclude_nmr \
            --delete_ligand $delete_ligand \
            --delete_protein $delete_protein"

echo "Evaluating Ensemble Model"
$config_ensemble
echo


cp "$0" $experiment_path