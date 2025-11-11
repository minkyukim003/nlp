import time
import os
import pandas as pd
import numpy as np
import torch as t
import json

from torch.utils.data import DataLoader
from DataLoader import MovieDataset
from DataPreprocesser import preprocess


#Sequence Flow
#1.   Preprocess raw data, output clean_data.csv.
#2.   Load clean_data.csv.
#2.5  Embed Glove6B to text. 
#3.   Train model through model.py (LSTM). (Build a checkpoint save and load in training loop).
#4.   Run the model and output results.

def _read_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    return df

def _build_csv(csv_file : str, df):
    df.to_csv(csv_file)

def preprocess_sequence():
    #Preprocessing.
    raw_train_fpath = "./data/training_raw_data.csv"
    raw_test_fpath = "./data/test_raw_data.csv"
    train_fpath = "./data/training_data.csv"
    test_fpath = "./data/test_data.csv"

    if os.path.exists(train_fpath) and os.path.exists(test_fpath):
        print("Run line 36. Both training and test files exist.")
    else:
        if not os.path.exists(train_fpath):
            print("Run line 39. Preprocessing train data.")
            raw_train_df = _read_csv(raw_train_fpath)
            train_df = preprocess(raw_train_df, True)
            _build_csv(train_fpath,train_df)

        if not os.path.exists(test_fpath):
            print("Run line 45. Preprocessing test data.")
            raw_test_df = _read_csv(raw_test_fpath)
            test_df = preprocess(raw_test_df, False)
            _build_csv(test_fpath, test_df)

def _read_glove(glove_path: str):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

def _get_embed(glove_path: str, token_idx, e_dim: int):
    vocab_size = len(token_idx)
    embeddings_matrix = np.zeros((vocab_size, e_dim))
    embeddings_index = _read_glove(glove_path)
    for word, idx in token_idx.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embeddings_matrix[idx] = vector
    return t.FloatTensor(embeddings_matrix)

def _embedding(pretrain: bool, glove_path: str, json_path: str, e_dim: int):
    if pretrain == True:
        with open(json_path,'r') as f:
            token_idx = json.load(f)
        embedding_matrix = _get_embed(glove_path, token_idx, e_dim)
    else:
        embedding_matrix = None
    return embedding_matrix

def main():
    #Preprocess.
    preprocess_sequence()

    #Set hyperparameters.
    mode = 'train'
    json_path = './data/token_idx.json'
    glove_path = "./glove.6B/glove.6B.50d.txt"
    glove_dim = 50
    batch_size_param = 300

    #Stage datasets.
    training_data = MovieDataset('./data/training_data.csv')
    test_data = MovieDataset('./data/test_data.csv')
    train_dataloader = DataLoader(
                                training_data, 
                                batch_size = batch_size_param, 
                                shuffle=True, 
                                num_workers=1
                            )
    test_dataloader = DataLoader(
                                test_data, 
                                batch_size = batch_size_param, 
                                shuffle=False, 
                                num_workers=1
                            )

    #Make embedding tensor. 
    embedding_matrix = _embedding(glove_path, json_path, glove_dim)

    #Prepare GPU/CPU for model execution. 
    use_cuda = t.cuda.is_available()
    device = t.device("cuda:0" if use_cuda else "cpu")
    #Ensure reproducibility.
    if use_cuda == True:
        t.cuda.manual_seed(9082)
    else:
        t.manual_seed(9082)

    

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Runtime: {start_time - end_time}.")
