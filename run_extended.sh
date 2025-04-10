#!/bin/bash

python -m experiments.readability_assessement_V2 --model_name FacebookAI/xlm-roberta-base --model_type xlmroberta --num_train_epochs 4 --run_mode word_file_cat --n_fold 3 --append_column word_file_cat
python -m experiments.readability_assessement_V2 --model_name FacebookAI/xlm-roberta-base --model_type xlmroberta --num_train_epochs 4 --run_mode word_file --n_fold 3 --append_column word_file
python -m experiments.readability_assessement_V2 --model_name google-bert/bert-base-multilingual-cased --model_type bert --num_train_epochs 4 --run_mode word_file_cat --n_fold 3 --append_column word_file_cat
python -m experiments.readability_assessement_V2 --model_name google-bert/bert-base-multilingual-cased --model_type bert --num_train_epochs 4 --run_mode word_file --n_fold 3 --append_column word_file
python -m experiments.readability_assessement_V2 --model_name aubmindlab/araelectra-base-discriminator --model_type electra --num_train_epochs 4 --run_mode word_file_cat --n_fold 3 --append_column word_file_cat
python -m experiments.readability_assessement_V2 --model_name aubmindlab/araelectra-base-discriminator --model_type electra --num_train_epochs 4 --run_mode word_file --n_fold 3 --append_column word_file
python -m experiments.readability_assessement_V2 --model_name aubmindlab/bert-base-arabertv2--model_type bert --num_train_epochs 4 --run_mode word_file_cat --n_fold 3 --append_column word_file_cat
python -m experiments.readability_assessement_V2 --model_name aubmindlab/bert-base-arabertv2 --model_type bert --num_train_epochs 4 --run_mode word_file --n_fold 3 --append_column word_file
python -m experiments.readability_assessement_V2 --model_name CAMeL-Lab/bert-base-arabic-camelbert-mix--model_type bert --num_train_epochs 4 --run_mode word_file_cat --n_fold 3 --append_column word_file_cat
python -m experiments.readability_assessement_V2 --model_name CAMeL-Lab/bert-base-arabic-camelbert-mix --model_type bert --num_train_epochs 4 --run_mode word_file --n_fold 3 --append_column word_file