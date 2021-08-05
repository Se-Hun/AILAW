import os
import argparse


def read_bio_txt(fn, sentence_splitter=None):
    sentence_list = []
    label_list = []
    with open(fn, 'r', encoding='utf-8') as f:
        sentence = ''
        labels = []
        for line in f:
            if ("\t" not in list(line)) and (line.lstrip().rstrip() != sentence_splitter):
                continue

            if line.lstrip().rstrip() == sentence_splitter:
                sentence_list.append(sentence)
                label_list.append(labels)

                sentence = ''
                labels = []
                continue
            sentence = sentence + line.split('\t')[0]
            labels.append(line.split('\t')[1].lstrip().rstrip())

    assert (len(sentence_list) == len(label_list)), "Number of sentences and number of labels are not matched."

    print("[TEXT] data is loaded from {} -- {}".format(fn, len(sentence_list)))
    return sentence_list, label_list

def build_vocab(fns):
    train_texts, train_labels = read_bio_txt(fns["input"]["train"], sentence_splitter="----")
    dev_texts, dev_labels = read_bio_txt(fns["input"]["dev"], sentence_splitter="----")
    test_texts, test_labels = read_bio_txt(fns["input"]["test"], sentence_splitter="----")

    # 2-d list --> 1-d list
    all_train_labels = sum(train_labels, [])
    all_dev_labels = sum(dev_labels, [])
    all_test_labels = sum(test_labels, [])

    # label coverage check
    _train_set = set(all_train_labels)
    _test_set = set(all_test_labels)
    _dev_set = set(all_dev_labels)

    # validation
    assert len(_test_set - _train_set) <= 0, "labels in test set are not in train"
    assert len(_dev_set - _train_set) <= 0, "labels in dev set are not in train"

    # building vocab
    label_vocab = [x for x in list(sorted(list(_train_set)))]

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
            "train" : os.path.join(data_folder, "train.bio.txt"),
            "dev" : os.path.join(data_folder, "dev.bio.txt"),
            "test" : os.path.join(data_folder, "dev.bio.txt")
        },
        "output": {
            "label_vocab" : os.path.join(data_folder, "label.vocab")
        }
    }

    build_vocab(fns)