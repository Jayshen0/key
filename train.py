# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:05:22 2020

@author: Ajie
"""


import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from get_loader import myDataset
from model import lstm
import torch.nn as nn
import torch.optim as optim
import torch


df = pd.read_csv('./Challenge_Data/1. Target Variables/Canada_Canola_Producer_Prices.csv')
df.iloc[:,8]=df.iloc[:,8].fillna(method='ffill')

data = df.iloc[:,8].values
data = data.reshape([-1,1])

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
data_scaled = sc.fit_transform(data)

train_x = []
train_y = []

for i in range(3,df.shape[0]):
    if df.iloc[i-3,2] != df.iloc[i,2]:
        continue
    
    train_x.append(data_scaled[i-3:i,0])
    train_y.append(data_scaled[i,0])


train = myDataset(train_x,train_y)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = lstm(train_x[0].shape[0])
model = model.to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=3*1e-3)


train_loader = DataLoader(train, batch_size=2,num_workers=2,shuffle=False)

epoch = 0
prev = float('inf')
while(True):

    tot_loss = 0
    for idx, data in enumerate(train_loader):
        x, label = data
        x = x.to(device)
        label = label.to(device)
        score = model.forward(x)
        loss = criterion(score, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss
    
    print(epoch,tot_loss)
    
    
    if tot_loss > prev:
        break
    
    prev_tot = tot_loss
    
torch.save(model.state_dict(), '\best_model.pkl')










