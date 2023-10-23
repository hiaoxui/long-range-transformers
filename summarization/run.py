import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task-name', required=True, type=str)
parser.add_argument("--segment-length", required=True, type=int)
parser.add_argument("--ckpt-index", required=True, type=int)
args = parser.parse_args()

CKPT_INDEX = args.ckpt_index
TASK_NAME = args.task_name
SEGMENT_LENGTH = args.segment_length

WARMUP_STEPS = 1000
LR = 5e-5

from rouge_score import rouge_scorer
from transformers import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pickle
import sys, os
import json

sys.path.append('../')
from ContextModels import Seq2Seq


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


def read_govreport():
    data = {}
    for mode in ["train", "valid"]:
        d = []
        with open('data/gov_report/{}.jsonl'.format(mode), 'r') as json_file:
            for line in json_file:
                line_obj = json.loads(line)
                cur = {}

                cur["tgt"] = line_obj["output"]
                cur["src"] = line_obj["input"]
                d.append(cur)
        data[mode] = d
    data["dev"] = data["valid"]
    data["test"] = data["valid"]
    return data


def read_sumscreen():
    data = {}
    for mode in ["train", "dev", "test"]:
        d = []
        with open("data/sumscreen/{}_{}.json".format(TASK_NAME, mode)) as fr:
            for line in tqdm(fr.readlines()):
                cur_d = json.loads(line)
                cur = {}

                cur["tgt"] = " ".join(cur_d["Recap"]).replace('@@ ', '')
                cur["src"] = " ".join(cur_d["Transcript"]).replace('@@ ', '')
                d.append(cur)

        data[mode] = d

    return data


def read_data():
    if TASK_NAME == "gov":
        return read_govreport()
    else:
        return read_sumscreen()


class BatchedDataset(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def process_data(tokenizer):
    if os.path.exists("cache-{}".format(TASK_NAME)):
        with open("cache-{}".format(TASK_NAME), 'rb') as fr:
            processed_data = pickle.load(fr)
        return processed_data

    data = read_data()

    eos_token = tokenizer.eos_token
    bos_token = tokenizer.bos_token
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    processed_data = ()
    for mode in ["train", "dev", "test"]:
        processed = []
        for d in tqdm(data[mode]):
            src = d["src"]
            tgt = d["tgt"]

            encoder_tokens = tokenizer.tokenize(src)
            if len(encoder_tokens) > 15000:
                encoder_tokens = encoder_tokens[:15000]
            input_ids = tokenizer.convert_tokens_to_ids([cls_token] + encoder_tokens + [sep_token])

            decoder_tokens = tokenizer.tokenize(tgt)
            if len(decoder_tokens) > 1020 and mode == "train":
                decoder_tokens = decoder_tokens[:1020]
            decoder_input_ids = tokenizer.convert_tokens_to_ids([bos_token] + decoder_tokens)
            decoder_label_ids = tokenizer.convert_tokens_to_ids(decoder_tokens + [eos_token])

            processed.append({"input_ids": np.array(input_ids),
                              "decoder_input_ids": np.array(decoder_input_ids),
                              "decoder_label_ids": np.array(decoder_label_ids)})
        processed_data = processed_data + (processed,)
    with open("cache-{}".format(TASK_NAME), 'wb') as fw:
        pickle.dump(processed_data, fw)
    return processed_data


def main():
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    data_train, data_dev, data_test = process_data(tokenizer)

    train_dataloader = DataLoader(dataset=BatchedDataset(data_train), pin_memory=True, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(dataset=BatchedDataset(data_test), pin_memory=True, batch_size=1)

    model = Seq2Seq.get_enecoder_decoder_model(max_length=SEGMENT_LENGTH)

    model.cuda()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = ReverseSqrtScheduler(optimizer, [LR], WARMUP_STEPS)

    epoch_start = 0
    if CKPT_INDEX != 0:
        checkpoint = torch.load("ckpt-{}-{}-{}".format(TASK_NAME, SEGMENT_LENGTH, CKPT_INDEX - 1))
        epoch_start = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _scheduler = checkpoint["scheduler"]
        scheduler.n_steps = _scheduler.n_steps
        del _scheduler
        del checkpoint

    for epoch in range(epoch_start, 100):
        model.train()
        all_loss = 0
        update_step = 0

        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].cuda()
            decoder_input_ids = batch["decoder_input_ids"].cuda()
            decoder_label_ids = batch["decoder_label_ids"].cuda()
            attention_mask = torch.ones(input_ids.size()).cuda()
            loss = model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                         decoder_attention_mask=None, labels=decoder_label_ids)["loss"]

            all_loss += loss.item()
            loss.backward()
            scheduler.step_and_update_lr()
            scheduler.zero_grad()
            torch.cuda.empty_cache()

            update_step += 1
            if update_step % 500 == 0:
                print("Update Steps {} loss: {}\n".format(update_step, all_loss / update_step))
            if update_step % 5000 == 0:
                with torch.no_grad():
                    model.eval()
                    evaluate(model, test_dataloader, tokenizer)
                    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(), "scheduler": scheduler},
                               "ckpt-{}-{}-{}".format(TASK_NAME, SEGMENT_LENGTH, CKPT_INDEX))
                model.train()
        print("epoch: {}, Update Steps {}, loss: {}\n".format(epoch, update_step, all_loss / update_step))
        with torch.no_grad():
            model.eval()
            evaluate(model, test_dataloader, tokenizer)
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "scheduler": scheduler}, "ckpt-{}-{}-{}".format(TASK_NAME, SEGMENT_LENGTH, CKPT_INDEX))


def evaluate(model, dataloader, tokenizer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    eos_token_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]

    targets = []
    preds = []

    scores = []
    rouge_1 = []
    rouge_2 = []
    rouge_l = []
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].cuda()
        decoder_label_ids = batch["decoder_label_ids"].tolist()
        output = model.generate(input_ids, max_length=1020)

        output = output.cpu().tolist()
        for i in range(len(output)):
            pred = output[i]
            index = pred[1:].index(eos_token_id) if eos_token_id in pred[1:] else -1
            pred_str = tokenizer.decode(pred[1:index + 1])

            target = decoder_label_ids[i]
            index = target.index(eos_token_id)
            target_str = tokenizer.decode(target[:index])

            score = scorer.score(target_str, pred_str)
            targets.append(target_str)
            preds.append(pred_str)
            rouge_1.append(score["rouge1"][2])  # "fmeasure"])
            rouge_2.append(score["rouge2"][2])  # "fmeasure"])
            rouge_l.append(score["rougeL"][2])  # "fmeasure"])
            scores.append(score)
    print("rouge_1: {}, rouge_2: {}, rouge_l: {}".format(sum(rouge_1) / len(rouge_1), sum(rouge_2) / len(rouge_2),
                                                         sum(rouge_l) / len(rouge_l)))


main()

