import os
import json

import pandas as pd

from common.utils import prepare_dir

def build_data(fns, mode):
    in_fn = fns["input"]
    to_fn = fns["output"][mode]

    sentences = []
    case_ids = []

    if mode == "classification":
        with open(in_fn, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for idx, ex in enumerate(data):
                case_id = ex["id"]
                ex_sentences = ex["sentences"]

                for sentence in ex_sentences:
                    sentences.append(sentence["text"])
                    case_ids.append(case_id)

    if mode == "ner":
        with open(in_fn, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for idx, ex in enumerate(data):
                case_id = ex["id"]
                ex_sentences = ex["tagged_sentences"]

                for sentence in ex_sentences:
                    sentences.append(sentence)
                    case_ids.append(case_id)

    # dump
    df = {
        "ID" : case_ids,
        "sentences" : sentences
    }
    df = pd.DataFrame(df)
    df.to_excel(to_fn)

def build_data_ner(fns):
    in_fn = fns["input"]
    to_fn = fns["output"]

    sentences = []
    case_ids = []
    with open(in_fn, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for idx, ex in enumerate(data):
            case_id = ex["id"]
            ex_sentences = ex["tagged_sentences"]

            for sentence in ex_sentences:
                sentences.append(sentence)
                case_ids.append(case_id)

    df = {
        "ID": case_ids,
        "sentences": sentences
    }
    df = pd.DataFrame(df)
    df.to_excel(to_fn)

if __name__ == '__main__':
    to_folder = os.path.join("./", "temp")
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join("./", "run", "law_ner_tag.json"),
        "output" : {
            "classification" : os.path.join(to_folder, "class_sentences.xlsx"),
            "ner" : os.path.join(to_folder, "ner_sentences.xlsx")
        }
        # "output": os.path.join(to_folder, "sentences.xlsx")
        # "output" : os.path.join(to_folder, "ner_sentences.xlsx")
    }

    build_data(fns, "classification") # for 관계 태깅
    build_data(fns, "ner") # for ner 태깅