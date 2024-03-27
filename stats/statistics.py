import pandas as pd

# dataset = pd.read_csv('data/dataset.csv')
#
# # print(dataset)
#
# groups = dataset.groupby('Grade').size()
# print(groups)

# Initial dataset
# data = pd.read_excel('data/readability/raw_data.xlsx')
# groups = data.groupby('Grade').size()
# print(groups)
#
# print("Stats for training set")
# data = pd.read_csv('data/readability/train.csv', sep='\t')
#
# print(f'training samples {len(data)}')
#
# groups = data.groupby('Grade').size()
# print(groups)
#
# print("Stats for test set")
# data = pd.read_csv('data/readability/test.csv', sep='\t')
# groups = data.groupby('Grade').size()
# print(groups)
#
# print("Stats for validation set")
# data = pd.read_csv('data/readability/validation.csv', sep='\t')
# groups = data.groupby('Grade').size()
# print(groups)

# print("Stats for all")
# data = pd.read_excel('data/readability/categorised/categorised_raw_data.xlsx')
# groups = data.groupby('Grade').size()
# print(groups)
#
#
# print("Stats for training set")
# data = pd.read_csv('data/readability/categorised/train.csv', sep='\t')
# groups = data.groupby('Grade').size()
# print(groups)
#
# print("Stats for test set")
# data = pd.read_csv('data/readability/categorised/test.csv', sep='\t')
# groups = data.groupby('Grade').size()
# print(groups)
#
# print("Stats for validation set")
# data = pd.read_csv('data/readability/categorised/validation.csv', sep='\t')
# groups = data.groupby('Grade').size()
# print(groups)

# Initial dataset
data = pd.read_csv('data/readability/extended/train.csv',sep='\t')
# data = pd.read_csv('data/readability/categorised/extended/test.csv',sep='\t')
groups = data.groupby('Arabic_Filename')

for key in groups.groups.keys():
    group_df = groups.get_group(key)
    sub_group = group_df.groupby('Grade').size()
    print(f'=================={key}============================')
    print(sub_group)