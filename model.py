import torch
import torch.nn as nn





class lstm(nn.Module):
    def __init__(self, feature_size, hidden_size=3, num_layers=1):
     
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, features):

        hiddens, _ = self.lstm(features.unsqueeze(1))
        outputs = self.linear(hiddens[0])
        return outputs

