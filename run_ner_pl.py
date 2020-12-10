import os
import argparse
import platform
from glob import glob

import numpy as np
from seqeval import metrics as seqeval_metrics

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

class NER(pl.LightningModule):
    def __init__(self,
                 task,
                 text_reader,
                 num_labels,
                 label_vocab,
                 learning_rate: float=2e-5):
        super().__init__()
        self.save_hyperparameters()

        # get label_voab
        self.label_vocab = label_vocab

        # prepare text reader
        from utils.readers import get_text_reader
        text_reader = get_text_reader(self.hparams.text_reader, self.hparams.task, num_labels)
        self.text_reader = text_reader

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids):
        outputs = self.text_reader(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask,
                                  labels=label_ids)

        return outputs # (loss, logits) --> logits : [batch_size, seq_length, num_labels]

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label_ids = batch

        loss, _ = self(input_ids, token_type_ids, attention_mask, label_ids)

        result = {"loss": loss}
        return result

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label_ids = batch

        loss, logits = self(input_ids, token_type_ids, attention_mask, label_ids)

        preds = torch.argmax(logits, dim=2)

        labels = label_ids
        result = {"loss": loss, "preds": preds, "labels": labels}
        return result

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        # remove padding
        out_label_list = [[] for _ in range(labels.shape[0])]
        preds_list = [[] for _ in range(preds.shape[0])]
        assert (len(out_label_list) == len(preds_list)), "Prediction and Label are not matched."

        from torch.nn import CrossEntropyLoss
        pad_token_label_id = CrossEntropyLoss().ignore_index

        label_map = {i: label for i, label in enumerate(list(self.label_vocab.keys()))}

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[labels[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        # metrics - F1
        val_f1 = seqeval_metrics.f1_score(out_label_list, preds_list)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label_ids = batch

        _, logits = self(input_ids, token_type_ids, attention_mask, label_ids)

        preds = torch.argmax(logits, dim=2)

        labels = label_ids
        result = {"preds": preds, "labels": labels}
        return result

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()

        # remove padding
        out_label_list = [[] for _ in range(labels.shape[0])]
        preds_list = [[] for _ in range(preds.shape[0])]
        assert(len(out_label_list) == len(preds_list)), "Prediction and Label are not matched."

        from torch.nn import CrossEntropyLoss
        pad_token_label_id = CrossEntropyLoss().ignore_index

        label_map = {i: label for i, label in enumerate(list(self.label_vocab.keys()))}

        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[labels[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        # metrics - Precision, Recall, F1
        result = {
            "precision": seqeval_metrics.precision_score(out_label_list, preds_list),
            "recall": seqeval_metrics.recall_score(out_label_list, preds_list),
            "f1": seqeval_metrics.f1_score(out_label_list, preds_list),
        }

        print()
        print(seqeval_metrics.classification_report(out_label_list, preds_list, digits=4))

        # 나중에 self.label_vocab을 이용해서 실제 태그로 바꾸고 text file에 예측 결과들 덤핑하는것도 짜야함!

        return result

    def configure_optimizers(self):
        from transformers import AdamW

        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        return parser


def main():
    pl.seed_everything(42) # set seed

    # Argument Setting -------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # mode specific --------------------------------------------------------------------------------
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to train text classifier.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to predict on real dataset.")

    # model specific -------------------------------------------------------------------------------
    parser.add_argument("--text_reader", help="bert, kobert, koelectra, others, ...", default="bert")

    # experiment settings --------------------------------------------------------------------------
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")  # bert has 512 tokens.
    parser.add_argument("--batch_size", help="batch_size", default=32, type=int)
    parser.add_argument("--gpu_id", help="gpu device id", default="0")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = NER.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # Dataset ----------------------------------------------------------------------------------------------------------
    from dataset import NER_Data_Module
    dm = NER_Data_Module("ner", args.text_reader, args.max_seq_length, args.batch_size)
    dm.prepare_data()
    # ------------------------------------------------------------------------------------------------------------------

    # Model Checkpoint -------------------------------------------------------------------------------------------------
    from pytorch_lightning.callbacks import ModelCheckpoint
    model_name = '{}'.format(args.text_reader)
    model_folder = './model/{}/{}'.format("ner", model_name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=model_folder,
                                          filename='{epoch:02d}-{val_loss:.2f}')
    # ------------------------------------------------------------------------------------------------------------------

    # Early Stopping ---------------------------------------------------------------------------------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        gpus=args.gpu_id if platform.system() != 'Windows' else 1,  # <-- for dev. pc
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback]
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Do train !
    if args.do_train:
        model = NER("ner", args.text_reader, dm.num_labels, dm.label_vocab)
        trainer.fit(model, dm)

    # Do predict !
    if args.do_predict:
        model_files = glob(os.path.join(model_folder, '*.ckpt'))
        best_fn = model_files[-1]
        model = NER.load_from_checkpoint(best_fn)
        trainer.test(model, test_dataloaders=[dm.test_dataloader()])

if __name__ == '__main__':
    main()