from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, BertForTokenClassification
from transformers import ElectraForSequenceClassification, ElectraForTokenClassification

def get_text_reader(reader_name, task, num_labels):
    # AILAW Corpus is korean dataset.
    # So, model is fixed to Korean Model such as multilingual-BERT, kobert, koelectra, etc.

    if reader_name == "bert":
        if task == "classification":
            model_name = "bert-base-multilingual-cased"
            text_reader = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        else: # ner
            model_name = "bert-base-multilingual-cased"
            text_reader = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    elif reader_name == "kobert":
        if task == "classification":
            model_name = "monologg/kobert"
            text_reader = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        else: # ner
            model_name = "monologg/kobert"
            text_reader = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    elif reader_name == "koelectra":
        if task == "classification":
            model_name = "monologg/koelectra-base-discriminator"
            text_reader = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        else: # ner
            model_name = "monologg/koelectra-base-discriminator"
            text_reader = ElectraForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    else:
        raise KeyError(reader_name)

    return text_reader

def get_tokenizer(reader_name):
    # AILAW Corpus is korean dataset.
    # So, tokenized is fixed to Korean Tokenizer such as multilingual-BERT tokenizer, kobert tokenizer, etc.

    if reader_name == "bert":
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif reader_name == "kobert":
        from utils.tokenization_kobert import KoBertTokenizer
        model_name = "monologg/kobert"
        tokenizer = KoBertTokenizer.from_pretrained(model_name)

    elif reader_name == "koelectra":
        from transformers import ElectraTokenizer
        model_name = "monologg/koelectra-base-discriminator"
        tokenizer = ElectraTokenizer.from_pretrained(model_name)

    else:
        raise KeyError(reader_name)

    return tokenizer
