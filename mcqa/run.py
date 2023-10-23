import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--segment-length", required=True, type=int)
parser.add_argument("--ckpt-index", required=True, type=int)
parser.add_argument("--use-global", default=False, action="store_true")

args = parser.parse_args()

CKPT_INDEX = args.ckpt_index
SEGMENT_LENGTH = args.segment_length
USE_GLOBAL = args.use_global

WARMUP_STEPS = 1000
LR = 5e-5

from transformers import AutoTokenizer, AdamW
import sys
import json

sys.path.append('../')
from ContextModels import Seq2Seq

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import pickle
import os
import torch.utils.data.distributed
import numpy as np



class MCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = Seq2Seq.get_enecoder_decoder_model(max_length=SEGMENT_LENGTH)

    def forward(self, input_ids, global_attention_mask, answer_ids, label_ids):

        attention_mask = torch.ones(input_ids.size()).cuda().long()

        if not USE_GLOBAL:
            global_attention_mask = None

        if self.training:
            loss = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask,
                              decoder_input_ids=answer_ids[label_ids][:, :-1], labels=answer_ids[label_ids][:, 1:])[
                "loss"]
            return loss

        res_losses = []
        for i in range(len(answer_ids)):
            loss = self.model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask,
                              decoder_input_ids=answer_ids[i][:, :-1], labels=answer_ids[i][:, 1:])["loss"]

            res_losses.append(loss.item())
        pred_index = np.argmin(np.array(res_losses))

        return 1 if pred_index == label_ids else 0


class MCDataset(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def extract_qa(string, target):
    question, string = string.split("\n (A)")
    answer_a, string = string.split("\n (B)")
    answer_b, string = string.split("\n (C)")
    answer_c, answer_d = string.split("\n (D)")
    if target.split() == answer_a.split():
        assert target.split() != answer_b.split()
        assert target.split() != answer_c.split()
        assert target.split() != answer_d.split()
        label_id = 0
    elif target.split() == answer_b.split():
        assert target.split() != answer_c.split()
        assert target.split() != answer_d.split()
        label_id = 1
    elif target.split() == answer_c.split():
        assert target.split() != answer_d.split()
        label_id = 2
    elif target.split() == answer_d.split():
        label_id = 3

    return question, [answer_a, answer_b, answer_c, answer_d], label_id


def process_data(tokenizer, mode="train"):
    print(mode)
    if os.path.exists("{}-cache".format(mode)):
        with open("{}-cache".format(mode), 'rb') as fr:
            data = pickle.load(fr)
    else:
        data = []
        with open('quality/{}.jsonl'.format(mode), 'r') as json_file:
            for line in tqdm(json_file):
                line_obj = json.loads(line)

                input_str = line_obj["input"]
                index = input_str.index("\n\n\n", input_str.index("\n (D)"))
                qa_string = input_str[:index]
                context = input_str[index:]

                question, [answer_a, answer_b, answer_c, answer_d], label_id = extract_qa(qa_string, line_obj["output"])

                qa_tokens = [tokenizer.sep_token] + tokenizer.tokenize(qa_string) + [tokenizer.sep_token]
                context_tokens = tokenizer.tokenize(context) + [tokenizer.sep_token]

                global_attention_mask = [1 for _ in range(len(qa_tokens))] + [0 for _ in range(len(context_tokens))]

                answer_a_tokens = [tokenizer.bos_token] + tokenizer.tokenize(answer_a) + [tokenizer.eos_token]
                answer_b_tokens = [tokenizer.bos_token] + tokenizer.tokenize(answer_b) + [tokenizer.eos_token]
                answer_c_tokens = [tokenizer.bos_token] + tokenizer.tokenize(answer_c) + [tokenizer.eos_token]
                answer_d_tokens = [tokenizer.bos_token] + tokenizer.tokenize(answer_d) + [tokenizer.eos_token]

                text_input_ids = tokenizer.convert_tokens_to_ids(qa_tokens + context_tokens)
                answer_a_input_ids = tokenizer.convert_tokens_to_ids(answer_a_tokens)
                answer_b_input_ids = tokenizer.convert_tokens_to_ids(answer_b_tokens)
                answer_c_input_ids = tokenizer.convert_tokens_to_ids(answer_c_tokens)
                answer_d_input_ids = tokenizer.convert_tokens_to_ids(answer_d_tokens)

                label_ids = [label_id]

                data.append(
                    [text_input_ids, global_attention_mask,
                     [answer_a_input_ids, answer_b_input_ids, answer_c_input_ids, answer_d_input_ids],
                     label_ids])

        with open("{}-cache".format(mode), 'wb') as fw:
            pickle.dump(data, fw)

    return data


class ReverseSqrtScheduler:
    def __init__(self, optimizer, lr, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

        self.decay_factor = [_lr * n_warmup_steps ** 0.5 for _lr in lr]
        self.lr_step = [(_lr - 0) / n_warmup_steps for _lr in lr]

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        self.n_steps += 1
        if self.n_steps < self.n_warmup_steps:
            lr = [self.n_steps * _lr for _lr in self.lr_step]
        else:
            lr = [_decay_factor * self.n_steps ** -0.5 for _decay_factor in self.decay_factor]

        for i, param_group in enumerate(self._optimizer.param_groups):
            param_group['lr'] = lr[i]


def main():
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

    train_dataloader = DataLoader(dataset=MCDataset(process_data(tokenizer, "train")), pin_memory=True,
                                  batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset=MCDataset(process_data(tokenizer, "validation")), pin_memory=True,
                                 batch_size=1)

    print(len(train_dataloader), len(test_dataloader))
    model = MCModel()
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = ReverseSqrtScheduler(optimizer, [LR], WARMUP_STEPS)

    epoch_start = 0
    if CKPT_INDEX != 0:
        checkpoint = torch.load("ckpt-{}-{}-{}".format(CKPT_INDEX - 1, SEGMENT_LENGTH, USE_GLOBAL))
        epoch_start = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = checkpoint["scheduler"]

    with torch.no_grad():
        model.eval()
        evaluate(model, test_dataloader)

    for epoch in range(epoch_start, 100):
        model.train()
        all_loss = 0
        update_step = 0

        for batch in tqdm(train_dataloader):
            input_ids, global_attention_mask, _answer_ids, label_ids = batch
            input_ids = torch.Tensor([input_ids]).cuda().long()
            global_attention_mask = torch.Tensor([global_attention_mask]).cuda().long()
            answer_ids = [torch.Tensor([_answer_id]).cuda().long() for _answer_id in _answer_ids]
            # label_ids = [label_ids]).cuda().long()
            loss = model(input_ids, global_attention_mask, answer_ids, label_ids[0])
            all_loss += loss.item()
            loss.backward()
            scheduler.step_and_update_lr()
            scheduler.zero_grad()
            update_step += 1
            if update_step % 100 == 0:
                print("Update Steps {} loss: {}\n".format(update_step, all_loss / update_step))

        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
        with torch.no_grad():
            model.eval()
            evaluate(model, test_dataloader)
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "scheduler": scheduler}, "ckpt-{}-{}-{}".format(CKPT_INDEX, SEGMENT_LENGTH, USE_GLOBAL))


def evaluate(model, dataloader):
    score = []

    for batch in tqdm(dataloader):
        input_ids, global_attention_mask, _answer_ids, label_ids = batch
        input_ids = torch.Tensor([input_ids]).cuda().long()
        global_attention_mask = torch.Tensor([global_attention_mask]).cuda().long()
        answer_ids = [torch.Tensor([_answer_id]).cuda().long() for _answer_id in _answer_ids]
        acc = model(input_ids, global_attention_mask, answer_ids, label_ids[0])

        score.append(acc)

    print("Accuracy Score: {}".format(sum(score) / len(score)))


main()
