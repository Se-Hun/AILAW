import os
import random
import argparse

import numpy as np

import torch

from common.utils import is_gpu_available, prepare_dir
from common.ml.hparams import HParams

def get_tokenizer(model_type, large_model, language):
    if model_type == 'bert':
        # tokenizer [ BERT x English ]
        if language == 'eng':
            from transformers import BertTokenizer
            if large_model:
                tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            else:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

                # tokenizer [ BERT x Korean ]
        if language == 'kor' or language == 'es' or language == 'th':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    return tokenizer

def select_model(hps):
    if hps.model_type == 'bert':
        if hps.lang_type == 'kor':
            from ner.network.token_classifier import BertMultilingualTokenClassifier as model
        if hps.lang_type == 'eng':
            from ner.network.token_classifier import BertEnglishTokenClassifier as model

    return model

def run_train(hps, fns):
    data_dir = fns["input"]
    to_fns = fns["output"]

    tokenizer = get_tokenizer(hps.model_type, hps.large_model, hps.lang_type)

    from ner.data.ner_dataset import NERDataset
    train_data = NERDataset(data_dir, tokenizer, max_seq_length=hps.max_seq_length, mode="train")
    dev_data = NERDataset(data_dir, tokenizer, max_seq_length=hps.max_seq_length, mode="dev")

    model = select_model(hps)
    model = model(hps)

    from ner.utils.train import train_ner as train
    train(model, train_data, to_fns, dev_data=dev_data)

def run_eval(fns):
    return None

# Main -----------------------------------------------------------------------------------------------------------------
def get_language_type():
    return 'kor'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # mode specific ----------------------------------------------------------------------
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to train ner tagger.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the test set.")

    # device -----------------------------------------------------------------------------
    parser.add_argument("--use_gpu", help="use gpu=True or False", default=True)
    parser.add_argument("--gpu_id", help="gpu device id", default="0")

    # model type -------------------------------------------------------------------------
    parser.add_argument("--model_type", help="lstm, bert, xlnet, others, ...", default="bert")  # bert fixed for now
    parser.add_argument("--large_model", action='store_true', help="Do you wanna bert-large model ?")

    # for training -----------------------------------------------------------------------
    parser.add_argument("--max_epoch", help="max_epoch", default=10)
    parser.add_argument("--batch_size", help="batch_size", default=50)
    parser.add_argument("--learning_rate", help="learning_rate", default=0.001)
    parser.add_argument("--seed", help='seed', type=int, default=42)

    args = parser.parse_args()
    # ------------------------------------------------------------------------------------

    # seed -----------------------------------------------------------------------------
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device ---------------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not is_gpu_available(): args.use_gpu = False

    # data directory -------------------------------------------------------------------
    data_dir = os.path.join("./", "data", "ner", "run")

    # model directory ------------------------------------------------------------------
    model_dir = os.path.join("./", "model", "ner")
    prepare_dir(model_dir)

    # language and seq length ---------------------------------------------------------
    lang_type = get_language_type()
    if lang_type == 'kor':
        max_seq_length = 40  # <-- fixed : for korean
    else:
        max_seq_length = 128  # <-- fixed : for english

    # fns ------------------------------------------------------------------------------
    fns = {
        "input" : data_dir,
        "output" : {
            "last" : os.path.join(model_dir, "model.out"),
            "best" : os.path.join(model_dir, "model.best.out"),
            "result" : os.path.join(model_dir, "result.txt")
        }
    }

    if args.do_train:
        hps = HParams(
            # model type
            model_type=args.model_type,
            large_model=args.large_model,
            lang_type=lang_type,

            # For Training
            max_epoch=int(args.max_epoch),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),

            # seq_length
            max_seq_length = max_seq_length,

            # use gpu
            use_gpu=args.use_gpu
        )
        hps.show()

        run_train(hps, fns)

    if args.do_eval:
        run_eval(fns)