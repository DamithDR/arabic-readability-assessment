from experiments.readability_assessement import get_data_frames
from utils.result_eval import print_information

with open('data/osman/coarse_pred.txt', 'r') as c_preds:
    preds_c = c_preds.readlines()
    preds_c = [pred.replace('\n', '') for pred in preds_c]

_, test_df_cat, _ = get_data_frames(file_path='data/readability/categorised/extended',
                                    append_column='word_file_cat')
test_df_cat['predictions'] = preds_c

with open('data/osman/fine_pred.txt', 'r') as f_preds:
    preds_f = f_preds.readlines()
    preds_f = [pred.replace('\n', '') for pred in preds_f]

_, test_df, _ = get_data_frames(file_path='data/readability/extended',
                                append_column='word_file')
test_df['predictions'] = preds_f

with open('osman_out_all.txt', 'w') as stat_f:
    w_recall, w_precision, w_f1, m_f1 = print_information(test_df, 'predictions', 'labels', stat_f)

with open('osman_out_cat_all.txt', 'w') as stat_f:
    w_recall, w_precision, w_f1, m_f1 = print_information(test_df_cat, 'predictions', 'labels', stat_f)