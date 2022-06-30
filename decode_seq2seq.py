"""BERT finetuning runner."""

import argparse
import json
import logging
import math
import os
import pickle
import random
from time import sleep

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import \
    BertTokenizer, RobertaTokenizer
from transformers.tokenization_bert import whitespace_tokenize

import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.modeling_decoding import LayoutlmForSeq2SeqDecoder, BertConfig
from s2s_ft.tokenization_minilm import MinilmTokenizer
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.utils import load_and_cache_layoutlm_examples, convert_src_layout_inputs_to_tokens, \
    get_tokens_from_src_and_index, convert_tgt_layout_inputs_to_tokens, load_and_cache_pure_text_examples
from s2s_tag.config import *
from s2s_tag.utils import *

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'minilm': MinilmTokenizer,
    'roberta': RobertaTokenizer,
    'unilm': UnilmTokenizer,
    'layoutlm': BertTokenizer,
}

class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")
    parser.add_argument("--meta_file", default=None, type=str)
    parser.add_argument("--next_token_mask", action='store_true')
    parser.add_argument("--extra_token_without_layout", action='store_true')
    parser.add_argument("--pure_text_dataset", action='store_true')

    parser.add_argument("--tokenizer_name", default=None, type=str, required=True,
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument("--input_folder", type=str, help="Input folder")
    parser.add_argument("--cached_feature_file", type=str)
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    args = parser.parse_args()
    
    update_extra_token(args.meta_file)    
    
    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    modify_tokenizer_vocab_file(args.tokenizer_name, force_new=True)
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        max_len=args.max_seq_length
    )
    tokenizer.add_special_tokens({'additional_special_tokens': SP_STRINGS_LIST})
    
    if args.model_type == "roberta":
        vocab = tokenizer.encoder
    else:
        vocab = tokenizer.vocab

    # NOTE: tokenizer cannot setattr, so move this to the initialization step
    # tokenizer.max_len = args.max_seq_length

    config_file = args.config_path if args.config_path else os.path.join(args.model_path, "config.json")
    logger.info("Read decoding config from: %s" % config_file)
    config = BertConfig.from_json_file(config_file)

    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
        list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        max_tgt_length=args.max_tgt_length, pos_shift=args.pos_shift,
        source_type_id=config.source_type_id, target_type_id=config.target_type_id,
        cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token,
        layout_flag=args.model_type == 'layoutlm',
    ))

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
        
    # NEXT TOKEN MASK
    if args.next_token_mask:
        next_token_mask_table = create_next_token_mask(args.meta_file, offset=args.max_seq_length - args.max_tgt_length)
        next_token_mask_table = torch.tensor(next_token_mask_table, dtype=torch.float32, device=device)
        next_token_mask_table = nn.Embedding.from_pretrained(next_token_mask_table)
    else:
        next_token_mask_table = None
        
    for model_recover_path in [args.model_path.strip()]:
        logger.info("***** Recover model: %s *****", model_recover_path)
        found_checkpoint_flag = True
        model = LayoutlmForSeq2SeqDecoder.from_pretrained(
            model_recover_path, config=config, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
            length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
            max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift,
            extra_token_with_layout=not args.extra_token_without_layout
        )

        if args.fp16:
            model.half()
            if next_token_mask_table is not None:
                next_token_mask_table.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length
        max_tgt_length = args.max_tgt_length

        example_path = args.input_file if args.input_file else args.input_folder

        if not args.pure_text_dataset:
            to_pred = load_and_cache_layoutlm_examples(
                example_path, tokenizer, local_rank=-1,
                cached_features_file=args.cached_feature_file, shuffle=False, layout_flag=args.model_type == 'layoutlm'
            )
        else:
            to_pred = load_and_cache_pure_text_examples(
                example_path, tokenizer, local_rank=-1,
                cached_features_file=args.cached_feature_file, shuffle=False
            )

        input_lines = convert_src_layout_inputs_to_tokens(to_pred, tokenizer.convert_ids_to_tokens, max_src_length,
                                                          layout_flag=args.model_type == 'layoutlm')

        # NOTE: add the sequence index through enumerate
        input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))

        total_batch = math.ceil(len(input_lines) / args.batch_size)

        os.makedirs(args.output_folder, exist_ok=True)
        result_out = os.path.join(args.output_folder, 'results.jsons')
        result_writer = open(result_out, "w", encoding="utf-8")
        
        result_meta = {'args': vars(args)}
        json.dump(result_meta, open(os.path.join(args.output_folder, 'meta.json'), 'w'), indent=4)

        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                batch_count += 1
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = model(input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv,
                                   next_token_mask_table=next_token_mask_table)
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                        
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        pred_idx = [x - 1 for x in w_ids]
                        result = {
                            'filename': to_pred[buf_id[i]]['filename'],
                            'part_idx': to_pred[buf_id[i]]['part_idx'],
                            'proc_data': to_pred[buf_id[i]],
                            'tokenized_data': buf[i],
                            'pred_idx': pred_idx
                        }
                        result_writer.write(json.dumps(result) + '\n')

                pbar.update(1)

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")


if __name__ == "__main__":
    main()
