# Arabic Readability Assessment

[DARES: Dataset for Arabic Readability Estimation of School Materials](https://aclanthology.org/2024.determit-1.10/) - Transformers based evaluation of readability of school materials

## Installation
You first need to install PyTorch. The recommended PyTorch version is 2.2.1
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) for more details specifically for the platforms.

When PyTorch has been installed, you can install requirements from source by cloning the repository and running:

```bash
git clone https://github.com/DamithDR/arabic-readability-assessment.git
cd arabic-readability-assessment
pip install -r requirements.txt
```

## Experiment Results
You can easily run experiments on DARES1.0 using following command and altering the parameters as you wish

```bash
python -m experiments.readability_assessement --model_name CAMeL-Lab/bert-base-arabic-camelbert-mix --model_type bert --num_train_epochs 4 --run_mode append_filename --n_fold 5 --cuda_device 0 --append_column Arabic_Filename
```

Similarly you can run experiments on DARES2.0 using following command and altering the parameters as you wish
```bash
python -m experiments.readability_assessement_V2 --model_name CAMeL-Lab/bert-base-arabic-camelbert-mix --model_type bert --num_train_epochs 4 --run_mode append_filename --n_fold 5 --cuda_device 0 --append_column Arabic_Filename
```

## Parameters
Please find the detailed descriptions of the parameters
```text
model_name              : Huggingface transformer model name that you need to experiment with; ex: google-bert/bert-base-multilingual-cased
model_type              : Type of the huggingface model; ex: bert
num_train_epochs        : Number of training epochs to train
run_mode                : The running mode ( described below )
n_fold                  : Number of runs per experiment
cuda_device             : The device number to run the experiment on
append_column           : The column name of the dataset to append

```

## Run Modes
```text
test                            : execution for testing purposes
balanced                        : balance dataset across all classes and run
categorised                     : run with categorised data
categorised_test                : run with categorised data for testing purposes
append_word                     : append the concept to the text 
append_word_categorised         : append concept to the categorised data experiment
append_filename                 : append subject to the text
append_filename_categorised     : append subject to the categorised data experiment
```

## Citation DARES1.0
```bash
@inproceedings{el-haj-etal-2024-dares,
    title = "{DARES}: Dataset for {A}rabic Readability Estimation of School Materials",
    author = "El-Haj, Mo  and
      Almujaiwel, Sultan  and
      Premasiri, Damith  and
      Ranasinghe, Tharindu  and
      Mitkov, Ruslan",
    editor = "Nunzio, Giorgio Maria Di  and
      Vezzani, Federica  and
      Ermakova, Liana  and
      Azarbonyad, Hosein  and
      Kamps, Jaap",
    booktitle = "Proceedings of the Workshop on DeTermIt! Evaluating Text Difficulty in a Multilingual Context @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.determit-1.10",
    pages = "103--113",
}
```

## Citation DARES2.0
Coming soon
