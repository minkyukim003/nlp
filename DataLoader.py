import torch as t
from torch.utils.data import Dataset
import pandas as pd
 
#This dataloader will take in a cleaned dataframe and turn it into tensors for PyTorch.
#Keep note that text is not automtically quantified and will need to go through embedding.
class MovieDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.df)
    
    #Get each element in a row 
    #and convert to Torch readable tensors 
    #*Except the string, which needs embedding. 
    def __getitem__(self, index):
        #Map 'neg' and 'pos' to 0 and 1 here. 
        label_map = {'neg':0, 'pos':1}

        row = self.df.iloc[index]
        content = row['Content']
        label = label_map[row['Label']]
        seq_len = row['seq_len']

        return content, t.tensor(label), t.tensor(seq_len)
