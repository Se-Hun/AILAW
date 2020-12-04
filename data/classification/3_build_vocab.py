import os

def build_vocab(fns):
    import pandas as pd
    to_train_fn = fns["input"]["train"]
    to_dev_fn = fns["input"]["dev"]
    train_df = pd.read_csv(fns["input"]["train"], sep='\t')
    # test_df = pd.read_csv(fns["input"]["test"], sep='\t')
    dev_df = pd.read_csv(fns["input"]["dev"], sep='\t')

    # label coverage check
    _train_set = set(train_df['label'].unique().tolist())
    # _test_set = set(test_df['label'].unique().tolist())
    _dev_set = set(dev_df['label'].unique().tolist())

    # validation
    # assert len(_test_set - _train_set) <= 0, "labels in test set are not in train"
    assert len(_dev_set - _train_set) <= 0, "labels in dev set are not in train"

    # building vocab
    label_vocab = [x for x in list(sorted(list(_train_set)))]

    # dumping data -- for removing header
    train_df.to_csv(to_train_fn, index=False, header=None, sep="\t")
    print("[Train] NER data is dumped at  ", to_train_fn)
    dev_df.to_csv(to_dev_fn, index=False, header=None, sep="\t")
    print("[Dev] NER data is dumped at  ", to_dev_fn)

    # dumping vocab file
    label_vocab_fn = fns["output"]["label_vocab"]
    with open(label_vocab_fn, 'w', encoding='utf-8') as f:
        for idx, label in enumerate(label_vocab):
            print("{}\t{}".format(label, idx), file=f)
        print("[Label] vocab is dumped at ", label_vocab_fn)


if __name__ == '__main__':
    data_folder = os.path.join("./", "run")

    fns = {
        "input": {
            "train" : os.path.join(data_folder, "train.tsv"),
            "dev" : os.path.join(data_folder, "dev.tsv"),
            # "test" : os.path.join(data_folder, "dev.bio.txt")
        },
        "output": {
            "label_vocab" : os.path.join(data_folder, "label.vocab")
        }
    }

    build_vocab(fns)