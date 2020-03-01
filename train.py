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

#harv = pd.read_csv('./Challenge_Data/2. Other Canola Production Data/Canada_Canola_Harvested_Area.csv')
#harv = harv[harv['Region'] == 'Alberta']

train_x = []
train_y = []

pre_x = defaultdict(list)
pre_y = []

for i in range(3,df.shape[0]):
    if df.iloc[i-3,2] != df.iloc[i,2]:
        continue
    
    train_x.append(data_scaled[i-3:i,0])
    train_y.append(data_scaled[i,0])
    
    if df.iloc[i,6] > '2018':
        pre_x[df.iloc[i,2]].append(data_scaled[i-3:i,0])
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

#model.load_state_dict(torch.load('best_model.pkl'))

while(True):
    epoch += 1
    tot_loss = 0
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
        
        rel_err += max(0,1-abs(float(label)-float(score)) / float(label))
    scheduler.step()
    print(epoch,rel_err/len(train_loader))
    
    
    if tot_loss > prev:
        break
    
    prev = tot_loss
    
torch.save(model.state_dict(), 'best_model.pkl')


sub = pd.read_csv('./Challenge_Data/to_be_filled.csv')

last = None
for i in range(sub.shape[0]):
    if sub.iloc[i,1][6] == '1':
        
        for idx, data in enumerate(pre_d[sub.iloc[i,0]]):
            x,label = data
            x = x.float()
            x = x.to(device)
            x = x.view([1,3])
            _ = model.forward(x)
       
        
        prev = []
        tmp = df[df['Region']==sub.iloc[i,0]]
        prev.append(float(tmp[tmp['Start Date']=='2018-10-01']['Value']))
        prev.append(float(tmp[tmp['Start Date']=='2018-11-01']['Value']))
        prev.append(float(tmp[tmp['Start Date']=='2018-12-01']['Value']))
        prev[0] = (prev[0]-min_v)/range_v
        prev[1] = (prev[1]-min_v)/range_v
        prev[2] = (prev[2]-min_v)/range_v
        
    else:
        prev = prev[1:] + [np.array([last])]
    
    print(prev)
    last = model.forward(torch.Tensor(prev).view([1,3]).to(device))
    last = float(last)
    sub.iloc[i,-1] = range_v*last + min_v
    
sub.to_csv('./solution.csv')











