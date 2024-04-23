###These codes are applied to evaluate the performance of submissions in the contest. This file is written by Dr. Shuo Zhang.
import numpy as np
import pandas as pd

def score(preds_path, labels_path):
    preds_raw = pd.read_csv(preds_path)
    preds_pd = preds_raw.sort_values(by=preds_raw.columns[0], ascending=True)
    pid = preds_pd.id.values
    preds = preds_pd.results.values

    labels_raw = pd.read_csv(labels_path)
    labels_pd = labels_raw.sort_values(by=labels_raw.columns[0], ascending=True)
    lid = labels_pd.id.values
    labels = labels_pd.results.values

    if (pid == lid).all():
        pass
    else:
        raise ImportError(
            "Incorrect ids, please check!!!"
        )

    correct_number = len(np.where(preds == labels)[0])
    total_number = len(preds)

    final_score = correct_number / total_number

    print('Your final score is {}'.format(final_score))

    return final_score
