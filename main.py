import time
import datetime
import os
import pandas as pd
import numpy as np
import torch as t
import json
import wandb

from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from DataLoader import MovieDataset
from DataPreprocesser import preprocess
from Model import LSTMModel
from sklearn import metrics


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

def preprocess_sequence(seq_len, token_idx_lim, min_len, max_len):
    #Preprocessing.
    raw_train_fpath = "./data/training_raw_data.csv"
    raw_test_fpath = "./data/test_raw_data.csv"
    train_fpath = "./data/training_data.csv"
    test_fpath = "./data/test_data.csv"

    if os.path.exists(train_fpath) and os.path.exists(test_fpath):
        print("Run line 36. Both training and test files exist.\n")
    else:
        if not os.path.exists(train_fpath):
            print("Run line 39. Preprocessing train data.")
            raw_train_df = _read_csv(raw_train_fpath)
            train_df = preprocess(raw_train_df, True, seq_len, token_idx_lim, min_len, max_len)
            _build_csv(train_fpath,train_df)

        if not os.path.exists(test_fpath):
            print("Run line 45. Preprocessing test data.")
            raw_test_df = _read_csv(raw_test_fpath)
            test_df = preprocess(raw_test_df, False, seq_len, token_idx_lim, min_len, max_len)
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
    #Set WandB
    wandb.init(project="NLP", name="Sentiment analysis with LSTM NLP.")

    #Set parameters.
    token_idx_lim = 8000
    min_len = 100
    max_len = 600

    json_path = './data/token_idx.json'
    glove_path = "./glove.6B/glove.6B.200d.txt"
    print(f"Using {glove_path}.\n")

    pretrain = True
    glove_dim = 200
    batch_size_param = 300

    hidden_size = 128
    num_layers = 1
    dropout_p = 0.3
    seq_len = 150

    train = True
    learning_rate = 0.002
    n_epochs = 10
    clip = 5

    #Preprocess. set_len, token_idx_lim, min_len, max_len
    preprocess_sequence(seq_len, token_idx_lim, min_len, max_len)
    print("Preprocess finished.\n")

    #Stage datasets.
    training_data = MovieDataset('./data/training_data.csv')
    test_data = MovieDataset('./data/test_data.csv')
    train_dataloader = DataLoader(
                                training_data, 
                                batch_size = batch_size_param, 
                                shuffle=True, 
                                num_workers=1
                            )
    print("Training data loaded.\n")
    test_dataloader = DataLoader(
                                test_data, 
                                batch_size = batch_size_param, 
                                shuffle=False, 
                                num_workers=1
                            )
    print("Test data loaded.\n")

    #Make embedding tensor. 
    embedding_matrix = _embedding(pretrain, glove_path, json_path, glove_dim)
    print("Created embedding matrix.\n")

    #Prepare GPU/CPU for model execution. 
    use_cuda = t.cuda.is_available()
    device = t.device("cuda:0" if use_cuda else "cpu")
    #Ensure reproducibility.
    if use_cuda == True:
        t.cuda.manual_seed(42)
    else:
        t.manual_seed(42)
    print("GPU/CPU has been set.")
    print(f"{device} is being used.\n")

    #Load model. self, embed_size, embed_matrix, hidden_size, num_layers, dropout_p, seq_len
    model = LSTMModel(glove_dim, embedding_matrix, hidden_size, num_layers, dropout_p, seq_len)
    model.to(device)
    print("Loaded the model.")

    #Adam and BCE
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    loss_fx = nn.BCELoss()
    print("Optimizer and loss function has been set.\n")

    #Training Loop.
    print("Beginning training...")
    if train == True:
        model.train()
        for epoch in range(n_epochs):
            for batch_inputs, batch_labels in train_dataloader:
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

                out = model(batch_inputs)
                out = out.squeeze(1)

                loss = loss_fx(out, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                label = batch_labels.cpu().detach().numpy()
                pred = t.round(out).cpu().detach().numpy()
                t_accuracy = metrics.accuracy_score(label, pred)

                wandb.log({'epoch': epoch, 'train_loss': loss.item(), 'train_accuracy':t_accuracy})

    #Evaluation loop
    preds = []
    labels = []
    print("Beginning testing...\n")
    with t.no_grad():
        model.eval()
        for batch_inputs, batch_labels in test_dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            out = model(batch_inputs)
            out = out.squeeze(1)

            labels.extend(batch_labels.cpu().numpy())
            preds.extend(t.round(out).cpu().numpy())

    accuracy = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds, average='macro')
    recall = metrics.recall_score(labels, preds, average='macro')
    f1 = metrics.f1_score(labels, preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Runtime: {end_time - start_time}s.")
    print(f"Current date and time is {datetime.datetime.now()}")
