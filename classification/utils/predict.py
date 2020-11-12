import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler

from classification.data.data_collator import default_data_collator

def prepare_eval(model, eval_data):
    # use gpu if set
    device = torch.device("cpu")  # default
    if torch.cuda.is_available():
        if model.hps.use_gpu:
            device = torch.device("cuda")
    if device.type == 'cuda':
        model = model.cuda()

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=model.hps.batch_size, collate_fn=default_data_collator)

    return model, device, eval_dataloader

def prepare_inputs(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    return inputs

def eval_classifier(model, eval_data, to_fns):
    model.eval()  # switch to eval mode

    model, device, eval_dataloader = prepare_eval(model, eval_data)

    with torch.no_grad():
        all_labels = []
        all_logits = []
        for step, inputs in enumerate(eval_dataloader):
            inputs = prepare_inputs(inputs, device)

            outputs = model(**inputs)

            labels = inputs["labels"]
            logits = outputs[1]

            print("{}/{}\r".format(step, len(eval_dataloader)), end="")
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
    print("Accuracy : {:.4f}".format(accuracy))

    # Saving Result
    to_result_fn = to_fns["result"]
    with open(to_result_fn, mode='w') as f:
        print("Accuracy", file=f)
        print("{:.4f}".format(accuracy), file=f)
    print("Evaluation Result file is dumped at ", to_result_fn)