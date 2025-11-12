from collections import Counter

import matplotlib as mp

import re
import json

#Remove any non-characters.
def _remove_punc(df):
    df['Cleaned'] = df['Content'].map(lambda x: re.sub(r'[^\w\s]',' ',x))
    return df

#Turn text into lowercase. 
def _lowercase(df):
    df['Cleaned'] = df['Cleaned'].map(lambda x: x.lower())
    return df

#Switch labels from strings to number. 
def _convert_labels(df):
    label_map = {'neg' : 0, 'pos' : 1}
    df['Label'] = df['Label'].map(lambda x: label_map[x])
    return df

#Re-compute length after removal of non-words. 
def _compute_seq_len(df):
    df['text_list'] = df['Cleaned'].map(lambda x: x.split())
    df['seq_len'] = df['text_list'].map(lambda x: len(x))
    return df

#Remove extremely short or long content.
def _remove_short(df, min_len):
    #Threshold = 100
    df = df[min_len <= df['seq_len']]
    return df

def _remove_long(df, max_len):
    #Threshold = 600
    df = df[max_len >= df['seq_len']]
    return df

#Tokenization
def _token_index(df, sig_range):
    # 1. Have all words put into a list to be tokenized. 
    # 2. Using "Counter", sort the list to find the most common token in the list.
    # 3. Make a vocab list as a dictionary with the most frequent word as the 2nd index and so on (0 and 1 being unk and pad).
    # 4. Then dump that dictionary into a json file. 
    words_series = df['text_list'].tolist()
    tokens = []
    for words in words_series:
        tokens.extend(words)

    tokens_counted = Counter(tokens)
    
    #Tokens sorted by frequency and determining the tokens that will be used for training.
    sorted_tokens = tokens_counted.most_common(len(tokens))
    sig_tokens = sorted_tokens[0:sig_range]

    #Assign index to sig_tokens by using a dictionary.
    token_idx = {'<pad>':0, '<unk>':1}

    start_key = 2
    for token, count in sig_tokens:
        token_idx[token] = start_key
        start_key += 1

    with open('./data/token_idx.json', 'w') as f:
        json.dump(token_idx, f, indent = 4)

    return token_idx

def _encode(x, token_idx: dict):
    encoded = []
    for word in x:
        encoded.append(token_idx.get(word, 1))
    return encoded

def _encoding(df, token_idx):
    df['Input'] = df['text_list'].map(lambda x: _encode(x, token_idx))
    return df

def _pad_or_trunc(x, seq_len):
    return x[:seq_len] + [0] * max(0, seq_len - len(x))

def _pad_trunc_df(df, seq_len):
    df['Input'] = df['Input'].map(lambda x: _pad_or_trunc(x, seq_len))
    return df

#Run functions. Use pandas dataframe as input. Return the clean dataframe. 
def preprocess(df, train : bool, set_len, token_idx_lim, min_len, max_len):
    df = _remove_punc(df)
    df = _lowercase(df)

    #TODO Implement _compute_seq_len(). 
    df = _compute_seq_len(df)
    df = _remove_short(df, min_len)
    df = _remove_long(df, max_len)

    df = _convert_labels(df)

    sig_range = token_idx_lim
    if train == True:
        token_idx = _token_index(df, sig_range)
    else:
        with open('./data/token_idx.json', 'r') as f:
            token_idx = json.load(f)

    df = _encoding(df, token_idx)

    seq_len = set_len
    df = _pad_trunc_df(df, seq_len)

    #print(df)
    return df