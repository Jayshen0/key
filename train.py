# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:05:22 2020

@author: Ajie
"""


import numpy as np
import pandas as pd
import torch.optim
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

min_v = min(data)
max_v = max(data)
range_v = max_v - min_v

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
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.7, last_epoch=-1)

train_loader = DataLoader(train, batch_size=1,num_workers=4,shuffle=False)

epoch = 0
prev = float('inf')
while(True):

    rel_err = 0
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, label = data
        x = x.float()
        x = x.to(device)
        label = label.to(device)
        score = model.forward(x)
        score = score.view([1])
        loss = criterion(score, label)
        loss.backward()
        optimizer.step()
        
        label = range_v*float(label) + min_v
        score = range_v*float(score) + min_v
        
        rel_err += abs(float(label)-float(score)) / float(label)
    scheduler.step()
    print(epoch,1-rel_err/len(train_loader))
    
    
    if rel_err > prev:
        break
    
    prev = rel_err
    
torch.save(model.state_dict(), '\best_model.pkl')










