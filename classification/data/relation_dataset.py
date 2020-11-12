import os

import torch
from torch.utils.data.dataset import Dataset

from classification.data.data_processor import RelationProcessor
from classification.data.utils import InputExample, InputFeatures

def convert_examples_to_features(examples, tokenizer, max_length, label_list=None, output_mode=None):
    if max_length is None:
        max_length = tokenizer.max_len

    processor = RelationProcessor()
    if label_list is None:
        label_list = processor.get_labels()
        print("Using label list {} for relation classification".format(label_list))
    if output_mode is None:
        output_mode = "classification"
        print("Using output mode {} for relation classification".format(output_mode))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample):
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        else:
            raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        print("*** Example ***")
        print("guid: {}".format(example.guid))
        print("features: {}".format(features[i]))

    return features


class RelationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    # args: GlueDataTrainingArguments
    # output_mode: str
    # features: List[InputFeatures]

    def __init__(self, data_dir, tokenizer, max_seq_length, mode):
        self.processor = RelationProcessor()
        self.output_mode = "classification"

        label_list = self.processor.get_labels()
        self.label_list = label_list

        if mode == "train":
            examples = self.processor.get_train_examples(data_dir)
        elif mode == "dev":
            examples = self.processor.get_dev_examples(data_dir)
        elif mode == "test":
            examples = self.processor.get_test_examples(data_dir)
        else:
            raise KeyError(mode)

        # for debugging
        # N = 70
        # examples = examples[:N]

        print("Creating features from dataset file at {}".format(data_dir))
        self.features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=label_list,
            output_mode=self.output_mode,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list