import torch as t
from torch import nn
import torch.autograd as autograd

import pandas as pd 
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, ):
        #Inherit __init__ from nn.Module for important functions. 
        super().__init__()

        self.lstm = nn.LSTM()

