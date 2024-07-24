#!/bin/bash

# Assign arguments to variables
model=$1

# Use the variables
echo "Argument 1: $ARG1"

python -m experiments.readability_assessement_V2 --model_name model --model_type bert --num_train_epochs 4 --run_mode raw --n_fold 3
python -m experiments.readability_assessement_V2 --model_name model --model_type bert --num_train_epochs 4 --run_mode raw_cat --n_fold 3
python -m experiments.readability_assessement_V2 --model_name model --model_type bert --num_train_epochs 4 --run_mode append_word --n_fold 3 --append_column Word
python -m experiments.readability_assessement_V2 --model_name model --model_type bert --num_train_epochs 4 --run_mode append_word_categorised --n_fold 3 --append_column Word
python -m experiments.readability_assessement_V2 --model_name model --model_type bert --num_train_epochs 4 --run_mode append_filename --n_fold 3 --append_column Arabic_Filename
python -m experiments.readability_assessement_V2 --model_name model --model_type bert --num_train_epochs 4 --run_mode append_filename_categorised --n_fold 3 --append_column Arabic_Filename