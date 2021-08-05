import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# Dataset Builders -----------------------------------------------------------------------------------------------------
class NERDataset(Dataset):
    def __init__(self, bio_data, tokenizer, label_vocab, max_seq_len):
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab

        self.max_seq_len = max_seq_len

        # save all case ids
        # self.case_ids = case_ids

        # for debugging -- to smallset
        # N = 70
        # bio_data = bio_data[:N]

        # get padding token label id
        from torch.nn import CrossEntropyLoss
        pad_token_label_id = CrossEntropyLoss().ignore_index

        # transform all data
        from utils.ner_collate import convert_examples_to_features_for_ner
        self.features = convert_examples_to_features_for_ner(bio_data,
                                                             label_vocab,
                                                             max_seq_len,
                                                             tokenizer,
                                                             pad_token_label_id=pad_token_label_id)

    def __getitem__(self, i):
        feature = self.features[i]
        # case_ids = self.case_ids[i]

        # Convert to Tensors and build dataset
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        label_ids = torch.tensor(feature.label_ids, dtype=torch.long)

        item = [input_ids, token_type_ids, attention_mask, label_ids]
        return item

    def __len__(self):
        return (len(self.features))

class ClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, label_vocab, max_seq_len):
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab

        self.max_seq_len = max_seq_len

        # for debugging -- to smallset
        # N = 70
        # df = df[:N]

        # transform all data
        from tqdm.auto import tqdm
        df_iterator = tqdm(df.iterrows(), desc="Iteration")

        self.texts = []
        self.labels = []
        for row_idx, (index, row) in enumerate(df_iterator):
            text = row[1]
            label_text = row[2]

            text_obj = tokenizer(text, padding='max_length', max_length=self.max_seq_len, truncation=True)
            label_id = self.label_vocab[label_text]

            self.texts.append(text_obj)
            self.labels.append(label_id)

    def __getitem__(self, i):
        input_ids = np.array(self.texts[i]['input_ids'])
        token_type_ids = np.array(self.texts[i]['token_type_ids'])
        attention_mask = np.array(self.texts[i]['attention_mask'])

        label_ids = np.array(self.labels[i])

        item = [input_ids, token_type_ids, attention_mask, label_ids]
        return item

    def __len__(self):
        return (len(self.texts))

class DocumentClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, label_vocab, max_seq_len):
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab

        self.max_seq_len = max_seq_len

        # for debugging -- to smallset
        # N = 70
        # df = df[:N]

        # transform all data
        from tqdm.auto import tqdm
        df_iterator = tqdm(df.iterrows(), desc="Iteration")

        self.texts = []
        self.labels = []
        for row_idx, (index, row) in enumerate(df_iterator):
            text = row[1]
            label_text = row[2]

            text_obj = tokenizer(text, padding='max_length', max_length=self.max_seq_len, truncation=True)
            label_id = self.label_vocab[label_text]

            self.texts.append(text_obj)
            self.labels.append(label_id)

    def __getitem__(self, i):
        input_ids = np.array(self.texts[i]['input_ids'])
        token_type_ids = np.array(self.texts[i]['token_type_ids'])
        attention_mask = np.array(self.texts[i]['attention_mask'])

        label_ids = np.array(self.labels[i])

        item = [input_ids, token_type_ids, attention_mask, label_ids]
        return item

    def __len__(self):
        return (len(self.texts))
# ----------------------------------------------------------------------------------------------------------------------

# Data Modules ---------------------------------------------------------------------------------------------------------
class NER_Data_Module(pl.LightningDataModule):
    def __init__(self, task, text_reader, max_seq_length, batch_size):
        super().__init__()

        self.task = task

        # prepare tokenizer
        from utils.readers import get_tokenizer
        self.tokenizer = get_tokenizer(text_reader)

        # data preparing params
        self.data_dir = os.path.join("./data", self.task, "run")
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        # number of labels for determining model's last dimension
        self.label_vocab = None
        self.num_labels = None

    def prepare_data(self):
        # vocab
        label_vocab = self._load_vocab(os.path.join(self.data_dir, "label.vocab"))
        self.label_vocab = label_vocab
        self.num_labels = len(label_vocab)

        # read data
        train_bio_data = self._read_txt(os.path.join(self.data_dir, "train.bio.txt"), sentence_splitter="----")
        valid_bio_data = self._read_txt(os.path.join(self.data_dir, "dev.bio.txt"), sentence_splitter="----")
        test_bio_data = self._read_txt(os.path.join(self.data_dir, "dev.bio.txt"), sentence_splitter="----")

        # building dataset
        dataset = task_to_dataset[self.task]

        self.train_dataset = dataset(train_bio_data, self.tokenizer, label_vocab, self.max_seq_length)
        self.valid_dataset = dataset(valid_bio_data, self.tokenizer, label_vocab, self.max_seq_length)
        self.test_dataset = dataset(test_bio_data, self.tokenizer, label_vocab, self.max_seq_length)

    def _load_vocab(self, fn):
        print("Vocab loading from {}".format(fn))

        vocab = {}
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                symbol, _id = line.split('\t')
                vocab[symbol] = int(_id)

        return vocab

    def _read_txt(self, fn, sentence_splitter=None):
        case_ids = []
        data = []
        with open(fn, 'r', encoding='utf-8') as f:
            sentence = ''
            labels = []

            for line in f:
                if ("\t" not in list(line)) and (line.lstrip().rstrip() != sentence_splitter):
                    case_ids.append(int(line.lstrip().rstrip()))
                    continue

                if line.lstrip().rstrip() == sentence_splitter:
                    data.append((sentence, labels))
                    sentence = ''
                    labels = []
                    continue
                sentence = sentence + line.split('\t')[0]
                labels.append(line.split('\t')[1].lstrip().rstrip())

        print("[Text] data is loaded from {} -- {}".format(fn, len(data)))
        return data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class Classification_Data_Module(pl.LightningDataModule):
    def __init__(self, task, text_reader, max_seq_length, batch_size):
        super().__init__()

        self.task = task

        # prepare tokenizer
        from utils.readers import get_tokenizer
        self.tokenizer = get_tokenizer(text_reader)

        # data preparing params
        self.data_dir = os.path.join("./data", self.task, "run")
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        # number of labels for determining model's last dimension
        self.label_vocab = None
        self.num_labels = None

    def prepare_data(self):
        # vocab
        label_vocab = self._load_vocab(os.path.join(self.data_dir, "label.vocab"))
        self.label_vocab = label_vocab
        self.num_labels = len(label_vocab)

        # read data
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.tsv"), sep='\t', header=None)
        valid_df = pd.read_csv(os.path.join(self.data_dir, "dev.tsv"), sep='\t', header=None)
        test_df = pd.read_csv(os.path.join(self.data_dir, "dev.tsv"), sep='\t', header=None)

        # building dataset
        dataset = task_to_dataset[self.task]

        self.train_dataset = dataset(train_df, self.tokenizer, label_vocab, self.max_seq_length)
        self.valid_dataset = dataset(valid_df, self.tokenizer, label_vocab, self.max_seq_length)
        self.test_dataset = dataset(test_df, self.tokenizer, label_vocab, self.max_seq_length)

    def _load_vocab(self, fn):
        print("Vocab loading from {}".format(fn))

        vocab = {}
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                symbol, _id = line.split('\t')
                vocab[symbol] = int(_id)

        return vocab

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
#-----------------------------------------------------------------------------------------------------------------------

# Map For Converting Task Name to Specific Dataset Builder -------------------------------------------------------------
# If you wanna add new task,
# (1) You should write dataset builder class.
# (2) Next, As follows, You should add task name(key) and dataset builder class name(value).
# Optionally, If you wanna add task of another types such as Sequence Labeling and Question & Answering,
# You should implement another LightningDataModules.

task_to_dataset = {
    "csie" : NERDataset,
    "csii" : ClassificationDataset,
    "doc_classification" : DocumentClassificationDataset
}
#-----------------------------------------------------------------------------------------------------------------------