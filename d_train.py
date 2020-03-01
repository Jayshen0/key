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


m = 'Alberta'
df = pd.read_csv('./Challenge_Data/1. Target Variables/Canada_Canola_Producer_Deliveries.csv')
df.iloc[:,8]=df.iloc[:,8].fillna(method='ffill')
df=df[df['Region'] == m]


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



month = 0
for i in range(24,df.shape[0]):
    
    month += 1
    
    if month == 13:
        month = 1
    cur = np.zeros([1,15],dtype=np.float32)
    cur[0][0] = data_scaled[i-1,0]
    cur[0][1] = data_scaled[i-2,0]
    cur[0][2] = data_scaled[i-3,0]
    cur[0][month+2] = 1
    train_x.append(cur)
    train_y.append(data_scaled[i,0]])
    
    
    
    


train = myDataset(train_x,train_y)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = lstm(1)
model = model.to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.7, last_epoch=-1)

train_loader = DataLoader(train, batch_size=1,num_workers=4,shuffle=False)

epoch = 0
prev = float('inf')

while(True):
    epoch += 1
    tot_loss = 0
    rel_err = 0
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, label = data
        x = torch.Tensor(x)
        x = x.double()
        print(x)
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
    
    
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, label = data
        x = x.to(device)
 
        label = label.to(device)
        score = model.forward(x)
        score = score.view([1])
        loss = criterion(score, label)
        tot_loss += float(loss)
    
    print(tot_loss)
    if tot_loss > prev:
        break
    
    if epoch == 30:
        break
        
    
    prev = tot_loss
    



sub = pd.read_csv('./Challenge_Data/to_be_filled.csv')



a = float(df[df['Start Date']=='2018-12-01']['Value'])
b = float(df[df['Start Date']=='2018-11-01']['Value'])
c = float(df[df['Start Date']=='2018-10-01']['Value'])

for i in range(12):

       
    cur = np.zeros([1,15])
    cur[0]=a
    cur[1]=b
    cur[2]=c
    cur[3+i] = 1
    last = model.forward(torch.Tensor(cur).view([1,15]).to(device))
    
    print(range_v*last + min_v)
    













