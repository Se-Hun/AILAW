import os
import csv

from ner.data.utils import InputExample

def ner_text_reader(fn, sentence_splitter=None):
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        sentence = ''
        labels = []
        for line in f:
            if line.lstrip().rstrip() == sentence_splitter:
                data.append((sentence, labels))
                sentence = ''
                labels = []
                continue
            sentence = sentence + line.split('\t')[0].lstrip().rstrip()
            labels.append(line.split('\t')[1].lstrip().rstrip())

        print("[Text] data is loaded from {} -- {}".format(fn, len(data)))
    return data

class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Gets an example from a dict with tensorflow tensors.
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """
        Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
        examples to the correct format.
        """
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_txt(cls, input_file, sentence_splitter=None):
        return ner_text_reader(input_file, sentence_splitter)

class NERProcessor(DataProcessor):
    """Processor for the NER data set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(os.path.join(data_dir, "train.bio.txt")))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "train.bio.txt"), sentence_splitter="----"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "dev.bio.txt"), sentence_splitter="----"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "test.bio.txt"), sentence_splitter="----"), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B-crime.when", "I-crime.when",  "B-crime.where", "I-crime.where", "B-crime.what", "I-crime.what",
                "B-victim.age", "I-victim.age", "B-weapon", "I-weapon", "B-injured.part", "I-injured.part",
                "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = None
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
