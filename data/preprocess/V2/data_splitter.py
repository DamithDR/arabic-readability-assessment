import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(path, filename):
    random_state = 777

    raw_data = pd.read_excel(f'{path}/{filename}')
    groups = raw_data.groupby("Fine-grained")
    grades = list(groups.groups.keys())

    train_df = pd.DataFrame(columns=["Word", "Fine-grained", "Coarse-grained", "Arabic_Filename", "Text", "num_sent"])
    test_df = pd.DataFrame(columns=["Word", "Fine-grained", "Coarse-grained", "Arabic_Filename", "Text", "num_sent"])
    validation_df = pd.DataFrame(columns=["Word", "Fine-grained", "Coarse-grained", "Arabic_Filename", "Text", "num_sent"])

    for grade in grades:
        grade_df = groups.get_group(grade)
        train_split, test_split = train_test_split(grade_df, test_size=0.3, random_state=random_state)
        test_split, validation_split = train_test_split(test_split, test_size=0.33, random_state=random_state)

        train_df = pd.concat([train_df, train_split], axis=0)
        test_df = pd.concat([test_df, test_split], axis=0)
        validation_df = pd.concat([validation_df, validation_split], axis=0)

    train_df.to_csv(f'{path}/train.csv', index=False, sep='\t')
    test_df.to_csv(f'{path}/test.csv', index=False, sep='\t')
    validation_df.to_csv(f'{path}/validation.csv', index=False, sep='\t')


if __name__ == '__main__':
    split_data('data/readability/data_V2.0', 'Paper_Dataset.xlsx')
