import os
import csv

import pandas as pd
from sklearn.model_selection import train_test_split

from common.utils import prepare_dir

def split_data(fns, seed):
    in_fn = fns["input"]
    to_train_fn = fns["output"]["train"]
    to_dev_fn = fns["output"]["dev"]

    df = pd.read_csv(in_fn, sep='\t')

    # getting all case id
    all_case_id = df["ID"].tolist()
    all_case_id = list(set(all_case_id))

    # split train-dev set
    train_case_ids, dev_case_ids = train_test_split(all_case_id, test_size=0.2, random_state=seed)
    train_case_ids = sorted(train_case_ids)
    dev_case_ids = sorted(dev_case_ids)

    # build dictionary for converting DataFrame
    train_data_dict = {
        "ID": [],
        "document": [],
        "relation": []
    }
    dev_data_dict = {
        "ID": [],
        "document": [],
        "relation": []
    }
    for case_id in train_case_ids:
        case = df[df["ID"] == case_id]
        train_data_dict["ID"].append(case["ID"].values[0])
        train_data_dict["document"].append(case["document"].values[0])
        train_data_dict["relation"].append(case["relation"].values[0])
    for case_id in dev_case_ids:
        case = df[df["ID"] == case_id]
        dev_data_dict["ID"].append(case["ID"].values[0])
        dev_data_dict["document"].append(case["document"].values[0])
        dev_data_dict["relation"].append(case["relation"].values[0])

    # Convert dictionary to DataFrame
    train_data = pd.DataFrame(train_data_dict)
    dev_data = pd.DataFrame(dev_data_dict)

    # dump all data
    train_data.to_csv(to_train_fn, index=False, header=True, sep="\t")
    print("[Train] Classification data is dumped at  ", to_train_fn)
    dev_data.to_csv(to_dev_fn, index=False, header=True, sep="\t")
    print("[Dev] Classification data is dumped at  ", to_dev_fn)

if __name__ == '__main__':
    to_folder = os.path.join("./", "run")
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join("./", "document_relation.tsv"),
        "output": {
            "train": os.path.join(to_folder, "train.tsv"),
            "dev": os.path.join(to_folder, "dev.tsv")
        }
    }

    seed = 42 # seed -- parallel!!!! for comparing sentence test set

    split_data(fns, seed)
