import torch as t
from torch import nn
import torch.autograd as autograd

class LSTMModel(nn.Module):
    def __init__(self, embed_size, embed_matrix, hidden_size, num_layers, dropout_p, seq_len):
        #Inherit __init__ from nn.Module for important functions. 
        super().__init__()

        #Embedding Parameter
        self.embed_matrix = embed_matrix

        #LSTM Parameters
        self.embed_size = embed_size #Glove embedding dimension. 
        self.hidden_size = hidden_size
        self.num_layers = num_layers 

        #Dropout Parameters
        self.dropout_p = dropout_p

        #Maxpool Parameters
        self.seq_len = seq_len

        #Fully connected layer will be linear. Linear Network Parameters
        self.out_size = 1

        #Create embedding tensors from embedding matrix (from Glove so freezing the weights).
        self.embed_tensor = nn.Embedding.from_pretrained(self.embed_matrix, freeze=True)

        #LSTM(input_size, hidden_size, num_layers, batch_first)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        #Output is hidden_size. 

        #Dropout(probability)
        self.dropout = nn.Dropout(self.dropout_p)

        #Maxpool1d(kernel_size, stride)
        self.maxpool = nn.MaxPool1d(self.seq_len)

        #FC Layer. Linear(in_features, out_features)
        self.fc = nn.Linear(self.hidden_size, self.out_size)

        #Sigmoid activation function.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        #Retrieve embedding tensor. 
        embeds = self.embed_tensor(x)

        #Run layers.
        #LSTM.  
        lstm_out, _ = self.lstm(embeds)

        #Maxpool.
        lstm_out = lstm_out.permute(0, 2, 1)
        pooled = self.maxpool(lstm_out)

        #Flatten
        output = t.flatten(pooled, 1)

        #Dropout
        output = self.dropout(output)

        #Linear.
        output = self.fc(output)

        #Sigmoid.
        output = self.sigmoid(output)

        return output