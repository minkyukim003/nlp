import time

import DataPreprocesser
import pandas as pd

from DataLoader import MovieDataset


#Sequence Flow
#1.   Preprocess raw data, output clean_data.csv.
#2.   Load clean_data.csv.
#2.5  Embed Glove6B to text. 
#3.   Train model through model.py (LSTM). (Build a checkpoint save and load in training loop).
#4.   Run the model and output results.

def read_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    return df

#TODO Implement Data Preprocessing


#TODO Implement Data Loader

#

def main():
    return 0

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"The run time was: {start_time - end_time}.")
