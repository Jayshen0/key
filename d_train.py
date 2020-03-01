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
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
import torch

m = 'Saskatchewan'
df = pd.read_csv('./Challenge_Data/1. Target Variables/Canada_Canola_Producer_Deliveries.csv')

df.iloc[:,8]=df.iloc[:,8].fillna(method='ffill')

data = []

for i in range(df.shape[0]):
    if df.iloc[i,2] == 'Alberta' and df['Start Date'].iloc[i] >'2006':
        data.append(df.iloc[i,8])
    
    if df.iloc[i,2] == 'Manitoba' and df['Start Date'].iloc[i] > '2007':
        data.append(df.iloc[i,8])
        
    
    if df.iloc[i,2] == m and df['Start Date'].iloc[i] > '2005':
        data.append(df.iloc[i,8])
        
        

min_v = min(data)
max_v = max(data)
range_v = max_v - min_v


for i in range(len(data)):
    data[i] = (data[i]-min_v) / range_v

#harv = pd.read_csv('./Challenge_Data/2. Other Canola Production Data/Canada_Canola_Harvested_Area.csv')
#harv = harv[harv['Region'] == 'Alberta']

train_x = []
train_y = []

pre_x = defaultdict(list)
pre_y = []

for i in range(3,df.shape[0]):
    if df.iloc[i-3,2] != df.iloc[i,2]:
        continue
    
    train_x.append(data_scaled[i-3:i])
    train_y.append(data_scaled[i])
    
    if df.iloc[i,6] > '2018':
        pre_x[df.iloc[i,2]].append(data_scaled[i-3:i])
        pre_y.append(0)


train = myDataset(train_x,train_y)
pre_d = dict()
for r in pre_x:
    pre_d[r] =DataLoader(myDataset(pre_x[r],pre_y), batch_size=1,num_workers=4,shuffle=False) 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = lstm(train_x[0].shape[0])
model = model.to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.7, last_epoch=-1)

train_loader = DataLoader(train, batch_size=1,num_workers=4,shuffle=False)

epoch = 0
prev = float('inf')
tot_loss = 0


while(True):
    epoch += 1
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
        
        tot_loss += float(loss)
        loss.backward()
        optimizer.step()
        
        label = range_v*float(label) + min_v
        score = range_v*float(score) + min_v
        
        rel_err += max(0,1-abs(float(label)-float(score)) / float(label))
    scheduler.step()
    print(epoch,rel_err/len(train_loader))
    
    
    if tot_loss > prev:
        break
    
    prev = tot_loss
   


torch.save(model.state_dict(), 'best_model_d.pkl')










