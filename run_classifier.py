import os
import random
import argparse

import numpy as np

import torch

from common.utils import prepare_dir, is_gpu_available, load_model
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
            from classification.network.classifier import BertMultilingualClassifier as model
        if hps.lang_type == 'eng':
            from classification.network.classifier import BertEnglishClassifier as model

    return model

def run_train(hps, fns):
    data_dir = fns["input"]
    to_fns = fns["output"]

    tokenizer = get_tokenizer(hps.model_type, hps.large_model, hps.lang_type)

    from classification.data.relation_dataset import RelationDataset
    train_data = RelationDataset(data_dir, tokenizer, max_seq_length=hps.max_seq_length, mode="train")
    dev_data = RelationDataset(data_dir, tokenizer, max_seq_length=hps.max_seq_length, mode="dev")

    model = select_model(hps)
    model = model(hps)

    from classification.utils.train import train_classifier as train
    train(model, train_data, to_fns, dev_data=dev_data)

def run_eval(fns):
    data_dir = fns["input"]
    to_fns = fns["output"]

    model_fn = to_fns["best"]
    model = load_model(model_fn)
    hps = model.hps

    tokenizer = get_tokenizer(hps.model_type, hps.large_model, hps.lang_type)

    from classification.data.relation_dataset import RelationDataset
    dev_data = RelationDataset(data_dir, tokenizer, max_seq_length=hps.max_seq_length, mode="dev")

    from classification.utils.predict import eval_classifier as eval
    eval(model, dev_data, to_fns)

    return None

# Main -----------------------------------------------------------------------------------------------------------------
def get_language_type():
    return 'kor'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # mode specific ----------------------------------------------------------------------
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to train classifier.")
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
    parser.add_argument("--keep_prob", help="keep prob. for dropout during training", default=0.1)

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
    data_dir = os.path.join("./", "data", "classification", "run")

    # model directory ------------------------------------------------------------------
    model_dir = os.path.join("./", "model", "classification")
    prepare_dir(model_dir)

    # language and seq length ---------------------------------------------------------
    lang_type = get_language_type()
    if lang_type == 'kor':
        max_seq_length = 40  # <-- fixed : for korean
    else:
        max_seq_length = 128  # <-- fixed : for english

    # fns ------------------------------------------------------------------------------
    fns = {
        "input": data_dir,
        "output": {
            "last": os.path.join(model_dir, "model.out"),
            "best": os.path.join(model_dir, "model.best.out"),
            "result": os.path.join(model_dir, "result.txt")
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
            use_gpu=args.use_gpu,

            # hidden layer
            hidden_size=768,
            num_labels=13, # 이거 계속 수정할 것!

            # for dropout
            keep_prob=args.keep_prob
        )
        hps.show()

        run_train(hps, fns)

    if args.do_eval:
        run_eval(fns)
