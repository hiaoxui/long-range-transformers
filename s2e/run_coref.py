from __future__ import absolute_import, division, print_function

import logging
import os
import shutil
import json
import torch
import re

from transformers import (
    AutoConfig, AutoTokenizer, CONFIG_MAPPING, LongformerConfig, RobertaConfig, XLNetConfig,
    BertConfig, BigBirdConfig
)


from modeling import S2E
from data import get_dataset
from cli import parse_args
from training import train, set_seed
from eval import Evaluator

import consts

logger = logging.getLogger(__name__)


def latest_checkpoint(cache_folder):
    max_step = -1
    if not os.path.exists(os.path.join(cache_folder, 'dev_out')):
        return None, 0
    for fn in os.listdir(os.path.join(cache_folder, 'dev_out')):
        rst = re.findall(r'checkpoint-(\d+)', fn)
        if len(rst) > 0:
            max_step = max(max_step, int(rst[0]))
    if max_step is None:
        return None, 0
    return os.path.join(cache_folder, 'dev_out', f'checkpoint-{max_step}'), max_step


def main():
    args = parse_args()

    if 'xlnet' in args.m:
        consts.SPEAKER_START = 31997
    if 'spanbert' in args.m:
        consts.SPEAKER_START = 28990

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    if args.s and os.path.exists(args.s) and os.listdir(args.s) and args.do_train and not args.f and not args.r:
        raise ValueError("Output directory ({}) already exists and is not empty".format(args.s))
    if args.f and os.path.isdir(args.s):
        assert not args.r
        shutil.rmtree(args.s)

    cache_path = args.s
    data_path = args.d
    os.makedirs(cache_path, exist_ok=True)
    os.makedirs(cache_path+'/data', exist_ok=True)
    os.makedirs(cache_path+'/tb', exist_ok=True)
    os.makedirs(data_path+'/pkl', exist_ok=True)
    sp = 'test' if args.do_test else 'dev'
    model_suffix = args.m.split('/')[-1]
    args.train_file = os.path.join(data_path, 'jsonlines', 'train.english.jsonlines')
    args.predict_file = os.path.join(data_path, 'jsonlines', f'{sp}.english.jsonlines')
    args.train_file_cache = os.path.join(data_path, 'pkl', f'train.{model_suffix}.pkl')
    args.predict_file_cache = os.path.join(data_path, 'pkl', f'{sp}.{model_suffix}.pkl')
    args.tensorboard_dir = os.path.join(cache_path, 'tensorboard')
    args.conll_path_for_eval = os.path.join(data_path, 'conll', f'{sp}.english.v4_gold_conll')
    args.output_dir = os.path.join(cache_path, f'{sp}_out')

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    with open(os.path.join(args.s, 'args.txt'), 'w') as f:
        f.write(str(args))

    for key, val in vars(args).items():
        logger.info(f"{key} - {val}")

    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, amp training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.amp)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    # if args.config_name:
    #     config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    # elif args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    # else:
    #     config = CONFIG_MAPPING[args.model_type]()
    #     logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer = AutoTokenizer.from_pretrained(args.m, use_fast=False)

    if 'longformer' in args.m:
        config_class = LongformerConfig
        base_model_prefix = "longformer"
    elif 'xlnet' in args.m:
        config_class = XLNetConfig
        base_model_prefix = "xlnet"
    elif 'bigbird' in args.m:
        config_class = BigBirdConfig
        base_model_prefix = 'bigbird'
    elif 'spanbert' in args.m or 'roberta' in args.m:
        config_class = BertConfig
        base_model_prefix = 'bert'
    else:
        raise NotImplementedError

    ckpt, start_step = latest_checkpoint(args.s)
    S2E.config_class = config_class
    S2E.base_model_prefix = base_model_prefix
    model = S2E(args)
    if args.pm is not None:
        model.load_state_dict(torch.load(args.pm), strict=False)
    if args.r:
        model.load_state_dict(torch.load(os.path.join(ckpt, 'pytorch_model.bin')))
    # model = S2E.from_pretrained(
    #     args.model_name_or_path,
    #     config=config,
    #     cache_dir=args.cache_dir,
    #     args=args,
    # )

    model.to(args.device)

    if args.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    evaluator = Evaluator(args, tokenizer)
    # Training
    if args.do_train:
        train_dataset = get_dataset(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, evaluator, args.r, ckpt, start_step)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {}

    if args.do_eval and args.local_rank in [-1, 0]:
        result = evaluator.evaluate(model, prefix="final_evaluation", official=True)
        with open(os.path.join(args.s, f'{sp}.conll.json'), 'w') as fp:
            json.dump(result, fp, indent=4)
        results.update(result)
        return results

    return results


if __name__ == "__main__":
    main()
