import torch as t
import pandas as pd

from torch.utils.data import Dataset
from ast import literal_eval
 
#This dataloader will take in a cleaned dataframe and turn it into tensors for PyTorch.
#Keep note that text is not automtically quantified and will need to go through embedding.
class MovieDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, converters={'Input': literal_eval})
    
    def __len__(self):
        return len(self.df)
    
    #Get each element in a row 
    #and convert to Torch readable tensors 
    #Strings would have been tokenized here. 
    def __getitem__(self, index):
        row = self.df.iloc[index]
        content = row['Input']
        label = row['Label']

        return t.tensor(content), t.tensor(label, dtype=t.float)
