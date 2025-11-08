import pandas as pd
import matplotlib as mp
import re

#Remove extremely short or long content.
def _remove_short(df):
    #Threshold = 100
    return 0

def _remove_long(df):
    #Threshold = 600
    return 0

#Remove any non-characters/whitespace.
def _remove_punc(df):
    df['Cleaned'] = df['Content'].map(lambda x: re.sub(r'[^\w\s]',' ',x))
    return df

#Turn text into lowercase. 
def _lowercase(df):
    df['Cleaned'] = df['Cleaned'].map(lambda x: str(x).lower())
    return df

#Run functions. Use pandas dataframe as input. Return a cleansed dataframe. 
def preprocess(df):
    _remove_punc(df)
    _lowercase(df)
    return df

def build_csv():
    return 0
