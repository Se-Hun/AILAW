import os
import csv

import pandas as pd
from sklearn.model_selection import train_test_split

from common.utils import prepare_dir

def split_data(fns, seed):
    in_fn = fns["input"]
    to_train_fn = fns["output"]["train"]
    to_dev_fn = fns["output"]["dev"]

    data = pd.read_csv(in_fn, sep='\t')

    split_case_id = (data.iloc[len(data) - 1]['ID'] // 10) * 8
    split_df_idx = data.loc[data['ID'] == split_case_id].index[0]

    train_data, dev_data = data[:split_df_idx], data[split_df_idx:]

    train_data.to_csv(to_train_fn, index=False, header=True, sep="\t")
    print("[Train] Classification data is dumped at  ", to_train_fn)
    dev_data.to_csv(to_dev_fn, index=False, header=True, sep="\t")
    print("[Dev] Classification data is dumped at  ", to_dev_fn)

if __name__ == '__main__':
    to_folder = os.path.join("./", "run")
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join("./", "relation.tsv"),
        "output": {
            "train": os.path.join(to_folder, "train.tsv"),
            "dev": os.path.join(to_folder, "dev.tsv")
        }
    }

    seed = 42

    split_data(fns, seed)
