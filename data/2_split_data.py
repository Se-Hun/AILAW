import os
import json

from sklearn.model_selection import train_test_split

from common.utils import prepare_dir

def split_data(fns, seed):
    in_fn = fns["input"]
    to_train_fn = fns["output"]["train"]
    to_test_fn = fns["output"]["test"]

    train_data = []
    test_data = []
    with open(in_fn, 'r', encoding='utf-8') as f:
        data = json.load(f)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

    with open(to_train_fn, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
        print("[Train] NER data is dumped at  ", to_train_fn)

    with open(to_test_fn, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
        print("[Train] NER data is dumped at  ", to_test_fn)


if __name__ == '__main__':
    to_folder = os.path.join("./", "run")
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join("./", "run", "law_ner_tag.json"),
        "output": {
            "train": os.path.join(to_folder, "train_ner_data.json"),
            "test": os.path.join(to_folder, "test_ner_data.json")
        }
    }

    seed = 42

    split_data(fns, seed)