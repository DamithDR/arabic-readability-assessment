#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d.dolamullage@lancaster.ac.uk
#SBATCH --output=/storage/hpc/41/dolamull/experiments/arabic-readability-assessment/output.log
#SBATCH --error=/storage/hpc/41/dolamull/experiments/arabic-readability-assessment/error.log

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/41/dolamull/conda_envs/llm_env
export HF_HOME=/scratch/hpc/41/dolamull/hf_cache

source <(grep -v '^#' .env | xargs -d '\n')

huggingface-cli login --token $HUGGINGFACE_TOKEN

python -m experiments.readability_assessement_V2 --model_name $1 --model_type $2 --num_train_epochs 4 --run_mode raw --n_fold 3
python -m experiments.readability_assessement_V2 --model_name $1 --model_type $2 --num_train_epochs 4 --run_mode raw_cat --n_fold 3
python -m experiments.readability_assessement_V2 --model_name $1 --model_type $2 --num_train_epochs 4 --run_mode append_word --n_fold 3 --append_column Word
python -m experiments.readability_assessement_V2 --model_name $1 --model_type $2 --num_train_epochs 4 --run_mode append_word_categorised --n_fold 3 --append_column Word
python -m experiments.readability_assessement_V2 --model_name $1 --model_type $2 --num_train_epochs 4 --run_mode append_filename --n_fold 3 --append_column Arabic_Filename
python -m experiments.readability_assessement_V2 --model_name $1 --model_type $2 --num_train_epochs 4 --run_mode append_filename_categorised --n_fold 3 --append_column Arabic_Filename