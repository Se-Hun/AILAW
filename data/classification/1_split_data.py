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

    # store all data at dictionary
    dict_about_case_id = {}
    for _idx, (index, row) in enumerate(df.iterrows()):
        case_id = row["ID"]
        case_sentence = row["sentences"]
        case_relation = row["relation"]

        if case_id not in list(dict_about_case_id.keys()):
            dict_about_case_id[case_id] = {"sentences" : [case_sentence], "relation" : [case_relation]}
        else:
            dict_about_case_id[case_id]["sentences"].append(case_sentence)
            dict_about_case_id[case_id]["relation"].append(case_relation)

    all_case_id = list(set(dict_about_case_id.keys()))

    # split train-dev set
    train_case_ids, dev_case_ids = train_test_split(all_case_id, test_size=0.2, random_state=seed)
    train_case_ids = sorted(train_case_ids)
    dev_case_ids = sorted(dev_case_ids)

    # build dictionary for converting DataFrame
    train_data_dict = {
        "ID": [],
        "sentences": [],
        "relation" : []
    }
    dev_data_dict = {
        "ID": [],
        "sentences": [],
        "relation": []
    }
    for case_id in train_case_ids:
        case_sentences = dict_about_case_id[case_id]["sentences"]
        case_relations = dict_about_case_id[case_id]["relation"]
        for case_sentence in case_sentences:
            train_data_dict["ID"].append(case_id)
            train_data_dict["sentences"].append(case_sentence)
        for case_relation in case_relations:
            train_data_dict["relation"].append(case_relation)
    for case_id in dev_case_ids:
        case_sentences = dict_about_case_id[case_id]["sentences"]
        case_relations = dict_about_case_id[case_id]["relation"]
        for case_sentence in case_sentences:
            dev_data_dict["ID"].append(case_id)
            dev_data_dict["sentences"].append(case_sentence)
        for case_relation in case_relations:
            dev_data_dict["relation"].append(case_relation)

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
        "input": os.path.join("./", "relation.tsv"),
        "output": {
            "train": os.path.join(to_folder, "train.tsv"),
            "dev": os.path.join(to_folder, "dev.tsv")
        }
    }

    seed = 42

    split_data(fns, seed)
