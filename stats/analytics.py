import pandas as pd

categorised_appended_filename = pd.read_csv('run_statistics/categorised_appended_filename.csv', sep='\t')
categorised_appended_word = pd.read_csv('run_statistics/categorised_appended_word.csv', sep='\t')

appended_filename = pd.read_csv('run_statistics/not_categorised_appended_filename.csv', sep='\t')
appended_word = pd.read_csv('run_statistics/not_categorised_appended_word.csv', sep='\t')

# print(appended_filename)
# print(appended_word)

improved_filenames = []

for file_prediction, word_prediction, filename, gold in zip(appended_filename['predictions'],
                                                            appended_word['predictions'],
                                                            appended_filename['Filename'], appended_filename['labels']):
    if file_prediction == gold and file_prediction != word_prediction:
        improved_filenames.append(filename)

print(len(improved_filenames))
print(improved_filenames)

improved_grades = [file.split('_')[1] for file in improved_filenames]
improved_grades_set = set(improved_grades)

print(improved_grades_set)