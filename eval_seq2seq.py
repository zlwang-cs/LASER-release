import argparse
import json
import os.path as op

import Levenshtein

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import BertTokenizer
from s2s_tag.config import *
from s2s_tag.utils import *

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

def extract_entities(seq):
    results = []

    p = q = r = 0
    while True:
        if p >= len(seq):
            break
        elif seq[p]['token'] == BEGIN_STRING:
            q = p + 1
            while True:
                if q >= len(seq):
                    break
                elif seq[q]['token'] == END_STRING:
                    r = q + 1
                    while True:
                        if r >= len(seq):
                            break
                        elif seq[r]['token'] == TAG_END_STRING:
                            sub_seq = list(filter(lambda x: x['idx_in_src'] != -1, seq[p+1:q]))
                            # sub_seq = seq[p+1:q]
                            tag_seq = list(filter(lambda x: x['idx_in_src'] == -1, seq[q+1:r]))
                            # tag_seq = seq[q+1:r]
                            results.append((sub_seq, tag_seq))
                            break
                        elif seq[r]['token'] in [BEGIN_STRING, END_STRING]:
                            break
                        r += 1
                    break
                elif seq[q] in EXTRA_TOKEN_LIST:
                    break
                q += 1
        p += 1

    return results


def refine_label(label, all_labels, label_rev_mapping):
    # return label
    dists = []
    for L in all_labels:
        dists.append((L, Levenshtein.distance(label, L)))
    dists = sorted(dists, key=lambda x: x[1])
    refined_label = dists[0][0]
    if label_rev_mapping is not None and refined_label in label_rev_mapping:
        refined_label = label_rev_mapping[refined_label]
    return refined_label


def get_seq_label_result(entities, src, all_labels, label_rev_mapping):
    idx_labels = {}
    for entity_info, label_info in entities:
        label = '.'.join(x['token'] for x in label_info)
        label = refine_label(label, all_labels, label_rev_mapping)
        entity_len = len(entity_info)
        for i, word in enumerate(entity_info):
            if entity_len == 1:
                idx_labels[word['idx_in_src']] = 'S-' + label
            else:
                if i == 0:
                    idx_labels[word['idx_in_src']] = 'B-' + label
                elif i == entity_len - 1:
                    idx_labels[word['idx_in_src']] = 'E-' + label
                else:
                    idx_labels[word['idx_in_src']] = 'I-' + label
    result = []
    for w in src:
        if w['idx_in_src'] in idx_labels:
            result.append(idx_labels[w['idx_in_src']])
        else:
            result.append('O')
    return result

def convert_ids_to_tokens(idxys, converter):
    ret = []
    idx_in_src = 0
    label_flag = False
    for idxy in idxys:
        token = converter(idxy[0])
        if token == END_STRING:
            label_flag = True
        elif token == TAG_END_STRING:
            label_flag = False
            
        if token in [BEGIN_STRING, END_STRING, TAG_END_STRING]:
            i = -1
        elif token in TAGS and label_flag:
            i = -1
        else:
            i = idx_in_src
            idx_in_src += 1
        ret.append({
                'idx_in_src': i,
                'token': token,
                'bbox': idxy[1:]
                })
    return ret

def get_pred_results(src, pred_idx, max_src_idx):
    pred = []
    for i in pred_idx:
        if i < max_src_idx:
            pred.append(src[i])
        else:
            ii = i - max_src_idx
            token = get_extra_token(ii)
            token_2d_pos = get_extra_token_2d_pos(token)
            pred.append(
                {
                'idx_in_src': -1,
                'token': token,
                'bbox': token_2d_pos
                }
            )
    return pred

def refine_pred_results(pred_idx, src_len, max_src_idx):
    new_pred_idx = []
    idx_set = set()
    for idx in pred_idx:
        if idx < max_src_idx:
            if idx < src_len:
                if idx not in idx_set:
                    new_pred_idx.append(idx)
                    idx_set.add(idx)
        else:
            new_pred_idx.append(idx)
    return new_pred_idx
    
def keep_first_token(seq):
    ret = []
    for x in seq:
        if not x['token'].startswith('##'):
            ret.append(x)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-result_folder', type=str)
    parser.add_argument('-dataset_folder', type=str)
    parser.add_argument('-label_mapping', type=str, default=None)
    args = parser.parse_args()
    
    update_extra_token(op.join(args.dataset_folder, 'meta.json'))
    
    result_meta = json.load(open(op.join(args.result_folder, 'meta.json')))
    dataset_meta = json.load(open(op.join(args.dataset_folder, 'meta.json')))
    if args.label_mapping is not None:
        label_mapping = json.load(open(args.label_mapping))
        label_rev_mapping = {v: k for k, v in label_mapping.items()}
    else:
        label_rev_mapping = None
    
    all_labels = dataset_meta['labels']

    max_src_idx = result_meta['args']['max_seq_length'] - result_meta['args']['max_tgt_length'] - 1
    
    tokenizer_path = result_meta['args']['tokenizer_name']
    modify_tokenizer_vocab_file(tokenizer_path, force_new=True)
    tokenizer = BertTokenizer.from_pretrained(
        tokenizer_path, do_lower_case=True,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': SP_STRINGS_LIST})

    # load results
    results = []
    with open(op.join(args.result_folder, 'results.jsons')) as fr:
        for line in fr:
            results.append(json.loads(line))
    
    seq_true = []
    seq_pred = []
    
    for result in results:
        filename = result['filename']
        part_idx = result['part_idx']
        data = result['proc_data']
        pred_idx = result['pred_idx']
        src_ids = data['source_ids']
        tgt_ids = data['target_ids']
        
        pred_idx = refine_pred_results(pred_idx, len(src_ids), max_src_idx)
        
        src = convert_ids_to_tokens(src_ids, tokenizer.convert_ids_to_tokens)
        tgt = convert_ids_to_tokens(tgt_ids, tokenizer.convert_ids_to_tokens)
        pred = get_pred_results(src, pred_idx, max_src_idx)
                
        src = keep_first_token(src)
        tgt = keep_first_token(tgt)
        pred = keep_first_token(pred)
                
        tgt_entities = extract_entities(tgt)
        pred_entities = extract_entities(pred)
         
        tgt_labels = get_seq_label_result(tgt_entities, src, all_labels, label_rev_mapping)
        pred_labels = get_seq_label_result(pred_entities, src, all_labels, label_rev_mapping)
        
        seq_true.append(tgt_labels)
        seq_pred.append(pred_labels)
    
    pre = precision_score(y_true=seq_true, y_pred=seq_pred)
    rec = recall_score(y_true=seq_true, y_pred=seq_pred)
    f1 = f1_score(y_true=seq_true, y_pred=seq_pred)
    report = classification_report(y_true=seq_true, y_pred=seq_pred)
    
    print('pre:', pre)
    print('rec:', rec)
    print('f-1:', f1)
    print(report)
    
    eval_result = {
        'pre': pre,
        'rec': rec,
        'f1': f1,
        'report': report,
    }
    with open(op.join(args.result_folder, 'evaluation.json'), 'w') as fw:
        json.dump(eval_result, fw, indent=4)

