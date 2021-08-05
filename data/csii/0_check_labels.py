import os

import pandas as pd

def check_labels(fn):
    data = pd.read_csv(fn, sep='\t')

    max_case_id = int(data.iloc[len(data) - 1]['ID'])

    pass_id = [80, 81, 85, 95]

    for id in range(1, max_case_id+1):
        if id in pass_id:
            continue

        df_idxs = data.loc[data['ID'] == id].index.tolist()

        target_df = data.iloc[df_idxs]
        labels = target_df.loc[target_df['relation'] != 13]
        labels = labels['relation'].tolist()

        label_check = set(labels)
        assert ((len(label_check) == 1) or len(label_check) == 0), "Another Label is composed to {} - {}".format(id, label_check)


if __name__ == '__main__':
    fn = os.path.join("./", "data.tsv")

    check_labels(fn)
