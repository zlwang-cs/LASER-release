import os
import json
import numpy as np
from s2s_tag.config import *


def modify_tokenizer_vocab_file(path, force_new=False):
    if os.path.exists(os.path.join(path, 'modified.tag')) and not force_new:
        return

    sp_token_replace_dict = {
        f'[unused{i}]': s for i, s in enumerate(SP_STRINGS_LIST)
    }
    lines = []
    original_vocab_file_path = os.path.join(path, 'vocab_original.txt')
    vocab_file_path = os.path.join(path, 'vocab.txt')
    with open(original_vocab_file_path) as fr:
        for line in fr:
            lines.append(sp_token_replace_dict.get(line.strip(), line.strip()))
    with open(vocab_file_path, 'w') as fw:
        for line in lines:
            fw.write(line + '\n')
    open(os.path.join(path, 'modified.tag'), 'w').write('modified')

    return


def update_extra_token(meta_path):
    global EXTRA_TOKEN_NUM
    
    meta = json.load(open(meta_path))
    tokens = meta['tokens']
    token_dict = meta['token_dict']
    
    for i, token in enumerate(tokens):
        TAGS.append(token)
        EXTRA_TOKEN_LIST.append(token)
        EXTRA_TOKEN_TOKENIZER_INDEX_DICT[token] = token_dict[token]
        
        v = len(EXTRA_TOKEN_PREDICT_BIAS_DICT)
        EXTRA_TOKEN_PREDICT_BIAS_DICT[token] = v
        # EXTRA_TOKEN_PREDICT_BIAS_REV_DICT[v] = token
        
        pos = 1003 + i + 1
        EXTRA_TOKEN_2D_POS_DICT[token] = [pos, pos, pos, pos]

get_extra_token_num = lambda : len(EXTRA_TOKEN_LIST)
get_extra_token_tokenizer_index = lambda token: EXTRA_TOKEN_TOKENIZER_INDEX_DICT[token]
get_extra_token_predict_bias = lambda token: EXTRA_TOKEN_PREDICT_BIAS_DICT[token]
get_extra_token_2d_pos = lambda token: EXTRA_TOKEN_2D_POS_DICT.get(token, [-1, -1, -1, -1])
get_extra_token = lambda i: EXTRA_TOKEN_LIST[i]

def get_few_shot_filenames(info_path, size, seed):
    if info_path is None:
        return None
    info = json.load(open(info_path))
    size = str(size)
    seed = str(seed)
    return info[size][seed]

def create_next_token_dict(meta):
    next_token_dict = {}
    
    assert set(meta['tokens']) == set(meta['words'])
    
    labels = meta['labels']
    for label in labels:
        label_words = label.split()
        label_words = [END_STRING] + label_words + [TAG_END_STRING]
        for i in range(len(label_words)-1):
            w = label_words[i]
            next_w = label_words[i+1]
            if w not in next_token_dict:
                next_token_dict[w] = []
            if next_w not in next_token_dict[w]:
                next_token_dict[w].append(next_w)
    
    return next_token_dict
    
def create_next_token_mask(meta_path, offset=513):
    # offset: max src len + 2(cls, sep)
    meta = json.load(open(meta_path))
    label_tokens = meta['tokens']
    next_token_dict = meta['next_token_dict']
    next_idx_dict = {}
    for k, vs in next_token_dict.items():
        # kid = get_extra_token_predict_bias(k) + max_src_len
        vids = [get_extra_token_predict_bias(v) + offset for v in vs]
        next_idx_dict[k] = vids
    cand_num = offset + get_extra_token_num()
    # 1: allow; 0: no
    mask = np.zeros((cand_num, cand_num))
    
    # normal words: [0, 511)
    # func:         [511, 514)
    # label words:  [514, ...
    
    # normal words [0, 511)
    mask[0:offset, 0:offset] = 1
    mask[0:offset, offset+get_extra_token_predict_bias(BEGIN_STRING)] = 1
    mask[0:offset, offset+get_extra_token_predict_bias(END_STRING)] = 1 
    
    # BEGIN 511
    mask[offset+get_extra_token_predict_bias(BEGIN_STRING), 0:offset] = 1
    
    # END 512
    end_next = next_idx_dict[END_STRING]
    mask[offset+get_extra_token_predict_bias(END_STRING), end_next] = 1
    
    # TAG END 513
    mask[offset+get_extra_token_predict_bias(TAG_END_STRING), 0:offset] = 1
    mask[offset+get_extra_token_predict_bias(TAG_END_STRING), offset+get_extra_token_predict_bias(BEGIN_STRING)] = 1
    
    # labels 514 ...
    for label in label_tokens:
        cur_id = offset+get_extra_token_predict_bias(label)
        next_ids = next_idx_dict[label]
        mask[cur_id, next_ids] = 1
    
    return mask

