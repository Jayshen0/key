import torch
import torch.nn as nn





class lstm(nn.Module):
    def __init__(self, feature_size, hidden_size=3, num_layers=1):
     
        super(lstm, self).__init__()
        self.linear = nn.Linear(15, 1)
        
    def forward(self, features):


        outputs = self.linear(features)
        return outputs

