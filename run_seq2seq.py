import argparse
import json
import logging
import os
import random
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler

from s2s_ft.configuration_layoutlm import LayoutlmConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import tqdm

from s2s_ft.modeling import LayoutlmForSequenceToSequence
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import \
    RobertaConfig, BertConfig, \
    BertTokenizer, RobertaTokenizer, \
    XLMRobertaConfig, XLMRobertaTokenizer

from s2s_ft.configuration_unilm import UnilmConfig
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.configuration_minilm import MinilmConfig
from s2s_ft.tokenization_minilm import MinilmTokenizer

from s2s_ft import utils
from s2s_ft.config import BertForSeq2SeqConfig

from s2s_tag.config import *
from s2s_tag.utils import *


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'minilm': (MinilmConfig, MinilmTokenizer),
    'roberta': (RobertaConfig, RobertaTokenizer),
    'xlm-roberta': (XLMRobertaConfig, XLMRobertaTokenizer),
    'unilm': (UnilmConfig, UnilmTokenizer),
    'layoutlm': (LayoutlmConfig, BertTokenizer),
}


def prepare_for_training(args, model, checkpoint_state_dict, amp):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        if checkpoint_state_dict:
            amp.load_state_dict(checkpoint_state_dict['amp'])

    if checkpoint_state_dict:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        model.load_state_dict(checkpoint_state_dict['model'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return model, optimizer


def train(args, training_features, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0] and args.log_dir:
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        tb_writer = None

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    # model recover
    recover_step = utils.get_max_epoch_model(args.output_dir)
    checkpoint_state_dict = None

    model.to(args.device)
    model, optimizer = prepare_for_training(args, model, checkpoint_state_dict, amp=amp)

    if args.n_gpu == 0 or args.no_cuda:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    else:
        per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps

    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    global_step = recover_step if recover_step else 0

    if args.num_training_steps == -1:
        args.num_training_steps = int(math.ceil(args.num_training_epochs * len(training_features) / train_batch_size))
        # args.num_training_steps = int(args.num_training_epochs * math.ceil(len(training_features) / train_batch_size))
    if args.num_warmup_steps == -1:
        args.num_warmup_steps = int(args.num_warmup_rate * args.num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps, last_epoch=-1)

    if checkpoint_state_dict:
        scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

    train_dataset = utils.Seq2seqDatasetForLayoutlm(
        features=training_features, max_source_len=args.max_source_seq_length,
        max_target_len=args.max_target_seq_length, vocab_size=tokenizer.vocab_size,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id, random_prob=args.random_prob, keep_prob=args.keep_prob,
        offset=train_batch_size * global_step, num_training_instances=train_batch_size * args.num_training_steps,
        layout_flag=args.model_type == 'layoutlm'
    )

    logger.info("Mode = %s" % str(model))

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_features))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.num_training_steps)

    if args.num_training_steps <= global_step:
        logger.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // args.gradient_accumulation_steps,
            collate_fn=utils.batch_list_to_batch_tensors)

        train_iterator = tqdm.tqdm(
            train_dataloader, initial=global_step,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])

        model.train()
        model.zero_grad()

        tr_loss, logging_loss = 0.0, 0.0

        for step, batch in enumerate(train_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'source_idxys': batch[0],
                      'target_idxys': batch[1],
                      'pseudo_idxys': batch[2],
                      'num_source_tokens': batch[3],
                      'num_target_tokens': batch[4],
                      'target_index': batch[-1],
                      }
            loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_last_lr()[0]))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.logging_steps == 0 and tb_writer is not None:
                    logging_loss = 0.0
                    tb_writer.add_scalar('train/lr', scheduler.get_lr()[0], global_step=global_step)
                    tb_writer.add_scalar('train/loss', loss.item(), global_step=global_step)

    if args.local_rank in [-1, 0] and args.save_last_step:
        save_path = os.path.join(args.output_dir, "ckpt-last")
        os.makedirs(save_path, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(save_path)
        logger.info("Saving model checkpoint last into %s", save_path)

    if args.local_rank in [-1, 0] and tb_writer:
        tb_writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str,
                        help="Training data (json format) for training. Keys: source and target")
    parser.add_argument("--train_folder", default=None, type=str,
                        help="Training data folder for training. Keys: source and target")
    parser.add_argument("--few_shot_info", default=None, type=str)
    parser.add_argument("--few_shot_size", default=None, type=int)
    parser.add_argument("--few_shot_seed", default=None, type=int)
    parser.add_argument("--meta_file", default=None, type=str)
    parser.add_argument("--extra_token_without_layout", action='store_true')
    parser.add_argument("--pure_text_dataset", action='store_true')
    
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=464, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=48, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Label smoothing.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument("--num_training_epochs", default=10, type=int,
                        help="set total number of training epochs to perform (--num_training_steps has higher priority)")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_warmup_rate", default=0.1, type=float,)

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")

    parser.add_argument('--logging_steps', type=int, default=-1,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_last_step', action='store_true')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    return args


def prepare(args):
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

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
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def get_model_and_tokenizer(args):
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model_config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length,
        max_source_length=args.max_source_seq_length,
        base_model_type=args.model_type,
        extra_token_with_layout=not args.extra_token_without_layout
    )

    logger.info("Model config for seq2seq: %s", str(config))

    if args.model_type == 'layoutlm':
        if args.tokenizer_name is not None:
            tokenizer_name = args.tokenizer_name
            modify_tokenizer_vocab_file(tokenizer_name, force_new=True)
        else:
            tokenizer_name = 'bert' + args.model_name_or_path[8:]
        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer.add_special_tokens({'additional_special_tokens': SP_STRINGS_LIST})
    else:
        tokenizer_name = args.tokenizer_name
        modify_tokenizer_vocab_file(tokenizer_name, force_new=True)
        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer.add_special_tokens({'additional_special_tokens': SP_STRINGS_LIST})

    model = LayoutlmForSequenceToSequence.from_pretrained(
        args.model_name_or_path, config=config, model_type=args.model_type,
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    return model, tokenizer


def main():
    args = get_args()
    update_extra_token(args.meta_file)
    few_shot_filenames = get_few_shot_filenames(args.few_shot_info, args.few_shot_size, args.few_shot_seed)
    
    prepare(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab
    # Load pretrained model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training will download model & vocab

    if args.cached_train_features_file is None:
        args.cached_train_features_file = os.path.join(args.output_dir, "cached_features_for_training.pt")

    example_path = args.train_file if args.train_file else args.train_folder
    
    if not args.pure_text_dataset:
        training_features = utils.load_and_cache_layoutlm_examples(
            example_path=example_path, tokenizer=tokenizer, local_rank=args.local_rank,
            cached_features_file=args.cached_train_features_file, max_src_length=args.max_source_seq_length,
            layout_flag=args.model_type == 'layoutlm', shuffle=True,
            white_list=few_shot_filenames
        )
    else:
        training_features = utils.load_and_cache_pure_text_examples(
            example_path=example_path, tokenizer=tokenizer, local_rank=args.local_rank,
            cached_features_file=args.cached_train_features_file, max_src_length=args.max_source_seq_length, shuffle=True,
            white_list=few_shot_filenames
        )

    train(args, training_features, model, tokenizer)


if __name__ == "__main__":
    main()
