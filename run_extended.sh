#!/bin/bash

python -m experiments.readability_assessement_V2 --model_name aubmindlab/bert-base-arabertv2 --model_type bert --num_train_epochs 4 --run_mode word_file_cat --n_fold 3 --append_column word_file_cat --lr 0.000005 --cuda_device 2
python -m experiments.readability_assessement_V2 --model_name CAMeL-Lab/bert-base-arabic-camelbert-mix --model_type bert --num_train_epochs 4 --run_mode word_file_cat --n_fold 3 --append_column word_file_cat --lr 0.000005 --cuda_device 2
python -m experiments.readability_assessement_V2 --model_name CAMeL-Lab/bert-base-arabic-camelbert-mix --model_type bert --num_train_epochs 4 --run_mode word_file --n_fold 3 --append_column word_file --lr 0.000005 --cuda_device 2