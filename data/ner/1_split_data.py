import os
import csv

import pandas as pd
from sklearn.model_selection import train_test_split

from common.utils import prepare_dir

def split_data(fns, seed):
    in_fn = fns["input"]
    to_train_fn = fns["output"]["train"]
    to_dev_fn = fns["output"]["dev"]

    train_data = []
    dev_data = []
    with open(in_fn, 'r', encoding="utf-8-sig") as f:
        data = pd.DataFrame(csv.reader(f, delimiter="\t"))

        train_data, dev_data = train_test_split(data, test_size=0.2, random_state=seed)

    train_data.to_csv(to_train_fn, index=False, header=None, sep="\t")
    print("[Train] NER data is dumped at  ", to_train_fn)
    dev_data.to_csv(to_dev_fn, index=False, header=None, sep="\t")
    print("[Dev] NER data is dumped at  ", to_dev_fn)

if __name__ == '__main__':
    to_folder = os.path.join("./", "run")
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join("./", "ner.tsv"),
        "output": {
            "train": os.path.join(to_folder, "train.tsv"),
            "dev": os.path.join(to_folder, "dev.tsv")
        }
    }

    seed = 42

    split_data(fns, seed)
