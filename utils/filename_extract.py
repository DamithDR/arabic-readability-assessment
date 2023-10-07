import pandas as pd


def append_column(read_file, write_file):
    data_file = pd.read_csv(read_file, sep='\t')

    file_names = data_file['Filename'].to_list()

    tags = list(set([f.split('_')[0] for f in file_names]))

    print(tags)
    print(len(tags))

    # manually found translations
    translated_word_dict = {
        'Psychology': 'علم النفس',
        'Economics': 'علم الاقتصاد',
        'History': 'التاريخ',
        'Ecology': 'علم البيئة',
        'Decision': 'صناعة القرار',
        'Health': 'الصحة',
        'Research': 'مهارات بحثية',
        'Business': 'الأعمال',
        'Physics': 'الفيزياء',
        'AI': 'الذكاء الاصطناعي',
        'Tech': 'التقنية الرقمية',
        'ArabicLanguage': 'اللغة العربية',
        'Athletics': 'اللياقة',
        'Software': 'برمجة',
        'Professional': 'التربية المهنية',
        'Math': 'الرياضيات',
        'Life&Family': 'المهارات الحياتية',
        'Artistic': 'الفنون',
        'Digital': 'المهارات الرقمية',
        'Life': 'المهارات الحياتية والأسرية',
        'Law': 'القانون',
        'Biology': 'الأحياء',
        'Computer': 'علوم الحاسب',
        'Geography': 'الجغرافيا',
        'Data': 'علم البيانات',
        'Geology': 'الجيولوجيا',
        'Sociology': 'علم الاجتماع',
        'Chemistry': 'الكيمياء',
        'Finance': 'الإدارة المالية',
        'Management': 'الإدارة',
        'Critical': 'التفكير الناقد',
        'IoT': 'إنترنت الأشياء',
        'Arabic': 'اللغة العربية',
        'Islamic': 'الدراسات الإسلامية',
        'Quran': 'علوم القرآن',
        'Hadith': 'الحديث',
        'Arts': 'التربية الفنية',
        'Science': 'العلوم'
    }

    arabic_file_names = [translated_word_dict.get(f.split('_')[0]) for f in file_names]

    print(len(arabic_file_names))
    print(len(file_names))

    data_file['Arabic_Filename'] = arabic_file_names
    data_file.to_csv(write_file, sep='\t', index=False)


if __name__ == '__main__':
    append_column(read_file='data/readability/train.csv', write_file='data/readability/extended/train.csv')
    append_column(read_file='data/readability/categorised/train.csv',
                  write_file='data/readability/categorised/extended/train.csv')
    append_column(read_file='data/readability/test.csv', write_file='data/readability/extended/test.csv')
    append_column(read_file='data/readability/categorised/test.csv',
                  write_file='data/readability/categorised/extended/test.csv')
    append_column(read_file='data/readability/validation.csv', write_file='data/readability/extended/validation.csv')
    append_column(read_file='data/readability/categorised/validation.csv',
                  write_file='data/readability/categorised/extended/validation.csv')

