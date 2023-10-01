import argparse
import os
import shutil

import numpy as np
import pandas as pd
import sklearn
import wandb
from simpletransformers.classification import ClassificationModel

import utils.arguments
from utils.label_encoder import encode, decode
from utils.result_eval import print_information


def get_data_frames():
    sampling_random_state = 777

    train_df = pd.read_csv('data/readability/train.csv', sep='\t')
    test_df = pd.read_csv('data/readability/test.csv', sep='\t')
    validation_df = pd.read_csv('data/readability/validation.csv', sep='\t')

    train_df.rename(columns={'Text': 'text', 'Grade': 'labels'}, inplace=True)
    test_df.rename(columns={'Text': 'text', 'Grade': 'labels'}, inplace=True)
    validation_df.rename(columns={'Text': 'text', 'Grade': 'labels'}, inplace=True)
    train_df = train_df.sample(frac=1, random_state=sampling_random_state)
    test_df = test_df.sample(frac=1, random_state=sampling_random_state)
    validation_df = validation_df.sample(frac=1, random_state=sampling_random_state)

    train_df['labels'] = encode(train_df['labels'].to_list())
    validation_df['labels'] = encode(validation_df['labels'].to_list())

    return train_df, test_df, validation_df


# Only for testing purposes
def get_test_data_frames():
    train_df, test_df, validation_df = get_data_frames()
    return train_df.head(500), test_df.head(100), validation_df.head(100)


def get_balanced_data_frames():
    train_df, test_df, validation_df = get_data_frames()
    # group by labels to limit the number
    return train_df, test_df, validation_df


def get_categorised_data_frames():
    train_df, test_df, validation_df = get_data_frames()
    # use categorised dataset and sample it frac = 1
    return train_df, test_df, validation_df


def get_training_arguments(args):
    training_arguments = utils.arguments.get_arguments()
    training_arguments.learning_rate = args.lr
    training_arguments.num_train_epochs = args.num_train_epochs
    training_arguments.train_batch_size = args.train_batch_size
    training_arguments.weight_decay = args.weight_decay
    return training_arguments


def run(args):
    if args.run_mode == "test":
        train_df, test_df, validation_df = get_test_data_frames()
    elif args.run_mode == "balanced":
        train_df, test_df, validation_df = get_balanced_data_frames()
    elif args.run_mode == "categorised":
        train_df, test_df, validation_df = get_categorised_data_frames()
    else:
        train_df, test_df, validation_df = get_data_frames()
    training_arguments = get_training_arguments(args)
    w_f1_lst = []
    m_f1_lst = []
    m_name = args.model_name.replace('/', '-')
    for i in range(args.n_fold):
        # remove and create an output dir
        if os.path.exists(training_arguments.output_dir) and os.path.isdir(training_arguments.output_dir):
            shutil.rmtree(training_arguments.output_dir)

        model = ClassificationModel(
            model_name=args.model_name, model_type=args.model_type, num_labels=len(set(train_df['labels'].to_list())),
            args=training_arguments,
            cuda_device=args.cuda_device
        )

        # Train the model
        model.train_model(train_df, eval_df=validation_df)

        # Predict
        predictions, raw_outputs = model.predict(test_df['text'].to_list())
        predictions = decode(predictions)
        test_df['predictions'] = predictions

        # write stats
        if args.run_stat_file is None:
            stat_file = f'run_statistics/stats_model_{m_name}_run_no_{i}'
        else:
            stat_file = f'run_statistics/stats_{args.run_stat_file}'

        with open(stat_file, 'w') as stat_f:
            labels = list(set(test_df['labels'].to_list())).sort()

            wandb.sklearn.plot_confusion_matrix(test_df['labels'].to_list(), predictions, labels=labels)
            stat_f.write('\n=========================\n')
            w_recall, w_precision, w_f1, m_f1 = print_information(test_df, 'predictions', 'labels', stat_f)
            w_f1_lst.append(w_f1)
            m_f1_lst.append(m_f1)
            wandb.log({'weighted_f1': w_f1, 'macro_f1': m_f1})
    with open(f'final_stats_{m_name}', 'w') as f:

        f.write(f'Weighted F1 mean: {np.mean(w_f1_lst)}| Weighted F1 STD: {np.std(w_f1_lst)}\n')
        wandb.log({'weighted_f1_mean': np.mean(w_f1_lst), 'weighted_f1_std': np.std(w_f1_lst)})

        f.write(f'Macro F1 mean: {np.mean(m_f1_lst)} | Macro F1 STD: {np.std(m_f1_lst)}\n')
        wandb.log({'macro_f1_mean': np.mean(m_f1_lst), 'macro_f1_std': np.std(m_f1_lst)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''evaluates models arabic readability assessment''')
    parser.add_argument('--model_name', required=True, help='model_name')
    parser.add_argument('--model_type', required=True, help='model_type')
    parser.add_argument('--lr', required=False, type=float, default=0.00005, help='learning_rate')
    parser.add_argument('--num_train_epochs', required=False, type=int, default=1, help='No of epochs')
    parser.add_argument('--train_batch_size', required=False, type=int, default=16, help='train_batch_size')
    parser.add_argument('--weight_decay', required=False, type=float, default=0, help='weight_decay')
    parser.add_argument('--cuda_device', required=False, type=int, default=0, help='cuda_device')
    parser.add_argument('--run_stat_file', required=False, help='run_stat_file')
    parser.add_argument('--n_fold', required=False, type=int, default=1, help='n_fold')
    parser.add_argument('--run_mode', required=False, default=False, help='run_mode')
    args = parser.parse_args()

    run(args)
