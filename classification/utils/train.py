import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from classification.data.data_collator import default_data_collator

def prepare_train(model, train_data, dev_data=None):
    # Optimizer ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Optimizer to use bert
    # to use bert
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    max_grad_norm = 1.0

    # use gpu if set
    device = torch.device("cpu")  # default
    if torch.cuda.is_available():
        if model.hps.use_gpu:
            device = torch.device("cuda")
    if device.type == 'cuda':
        model = model.cuda()

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=model.hps.batch_size, collate_fn=default_data_collator)

    if dev_data is not None:
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=model.hps.batch_size, collate_fn=default_data_collator)

    return model, device, optimizer, train_dataloader, dev_dataloader

def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    return inputs

def do_inner_eval(model, device, dev_dataloader):
    model.eval()  # switch to eval mode

    with torch.no_grad():
        all_labels = []
        all_logits = []
        for step, inputs in enumerate(dev_dataloader):
            inputs = prepare_inputs(inputs, device)

            outputs = model(**inputs)

            labels = inputs["labels"]
            logits = outputs[1]

            print("{}/{}\r".format(step, len(dev_dataloader)),end="")
            all_labels.append(labels)
            all_logits.append(logits)
    print()

    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    all_labels = all_labels.cpu().numpy().tolist()

    all_preds = []
    for logit in all_logits:
        best_id = torch.argmax(logit).item()
        all_preds.append(best_id)

    corrects = []
    for r, p in zip(all_labels, all_preds):
        if r == p:
            corrects.append(True)
        else:
            corrects.append(False)

    accuracy = np.mean(corrects)

    model.train()  # switch back to train mode
    return model, accuracy

def train_classifier(model, train_data, to_fns, dev_data=None):
    model.train()

    model, device, optimizer, train_dataloader, dev_dataloader = prepare_train(model, train_data, dev_data=dev_data)

    best_model_fn = to_fns["best"]

    _avg_losses = []
    max_score = -999
    for epoch in range(int(model.hps.max_epoch)):
        if dev_dataloader != None:
            model, acc = do_inner_eval(model, device, dev_dataloader)

            if acc > max_score:
                max_score = acc
                # save best model
                print("Best Model Updated!!")
                torch.save(model, best_model_fn)
                print("Best Model Updated!! .... Model saved at {} -- acc_and_f1 : {:6.4f}".format(best_model_fn, max_score))

        for step, inputs in enumerate(train_dataloader):
            inputs = prepare_inputs(inputs, device)

            outputs = model(**inputs)

            loss = outputs[0]
            loss.backward()

            _avg_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                print("Epoch : {} Step : {} Loss : {}".format(
                    epoch+1, step, np.mean(_avg_losses)
                ))
                _avg_losses = []

    # save model
    to_model_fn = to_fns["last"]
    torch.save(model, to_model_fn)
    print("Model saved at {}".format(to_model_fn))