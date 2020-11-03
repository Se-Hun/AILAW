import os
import random
import argparse

import numpy as np

import torch

from common.utils import is_gpu_available

def run_train():
    return None

def run_eval():
    return None

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