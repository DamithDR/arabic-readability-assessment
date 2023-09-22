import pandas as pd

# dataset = pd.read_csv('data/dataset.csv')
#
# # print(dataset)
#
# groups = dataset.groupby('Grade').size()
# print(groups)

# Initial dataset
data = pd.read_excel('data/readability/raw_data.xlsx')
groups = data.groupby('Grade').size()
print(groups)

print("Stats for training set")
data = pd.read_csv('data/readability/train.csv',sep='\t')
groups = data.groupby('Grade').size()
print(groups)

print("Stats for test set")
data = pd.read_csv('data/readability/test.csv',sep='\t')
groups = data.groupby('Grade').size()
print(groups)

print("Stats for validation set")
data = pd.read_csv('data/readability/validation.csv',sep='\t')
groups = data.groupby('Grade').size()
print(groups)