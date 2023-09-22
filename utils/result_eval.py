from sklearn.metrics import recall_score, precision_score, f1_score


def print_information(df, pred_column, real_column, output):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    labels = list(set(real_values))

    for label in labels:
        output.write('\noutput for labels\n')
        output.write(f"Stat of the {label} Class ")
        output.write(
            "\nRecall {}".format(
                recall_score(real_values, predictions, labels=labels, pos_label=label, average='macro')))
        output.write("\nPrecision {}".format(
            precision_score(real_values, predictions, labels=labels, pos_label=label, average='macro')))
        output.write(
            "\nF1 Score {}".format(f1_score(real_values, predictions, labels=labels, pos_label=label, average='macro')))

    output.write('\n====================\n')
    w_recall = recall_score(real_values, predictions, average='weighted')
    w_precision = precision_score(real_values, predictions, average='weighted')
    w_f1 = f1_score(real_values, predictions, average='weighted')
    m_f1 = f1_score(real_values, predictions, average='macro')

    output.write("\nWeighted Recall {}".format(w_recall))
    output.write("\nWeighted Precision {}".format(w_precision))
    output.write("\nWeighted F1 Score {}".format(w_f1))
    output.write("\nMacro F1 Score {}".format(m_f1))

    return w_recall, w_precision, w_f1, m_f1
