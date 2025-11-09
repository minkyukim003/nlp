import time
import pandas as pd

from DataLoader import MovieDataset
from DataPreprocesser import preprocess


#Sequence Flow
#1.   Preprocess raw data, output clean_data.csv.
#2.   Load clean_data.csv.
#2.5  Embed Glove6B to text. 
#3.   Train model through model.py (LSTM). (Build a checkpoint save and load in training loop).
#4.   Run the model and output results.

def read_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    return df

def build_csv(csv_file : str, df):
    df.to_csv(csv_file)


#TODO Implement Data Loader

#

def main():
    #Preprocessing.
    raw_train_fpath = "./data/training_raw_data.csv"
    raw_test_fpath = "./data/test_raw_data.csv"
    train_fpath = "./data/training_data.csv"
    test_fpath = "./data/test_data.csv"

    raw_train_df = read_csv(raw_train_fpath)
    raw_test_df = read_csv(raw_test_fpath)

    train_df = preprocess(raw_train_df, True)
    test_df = preprocess(raw_test_df, False)

    build_csv(train_fpath,train_df)
    build_csv(test_fpath, test_df)

    return 0

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Runtime: {start_time - end_time}.")
