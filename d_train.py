# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:05:22 2020

@author: Ajie
"""


import numpy as np
import pandas as pd


df = pd.read_csv('./Challenge_Data/1. Target Variables/Canada_Canola_Producer_Deliveries.csv')
df.iloc[:,8]=df.iloc[:,8].fillna(method='ffill')

df = df[df['Start Date'] > '2010']

d = dict()

d['British Columbia'] = [1 for i in range(11)]

for region in ['Alberta','Manitoba','Ontario','Québec','Saskatchewan']:
    cur = df[df['Region']==region]
    now = [0 for i in range(11)]
    for i in range(2011,2020):
        tmp = cur[cur['Start Date'] < str(i)]
        cur = cur[cur['Start Date'] > str(i)]
        tmp = list(tmp['Value'].values)
        

        for j in range(11):
            now[j] += tmp[j+1] / tmp[j]
    
    for k in range(11):
        now[k] /= 9
    
    d[region] = now
            




sub = pd.read_csv('./solution.csv')

i = 0
for r in ['Alberta','British Columbia', 'Manitoba', 'Ontario','Québec','Saskatchewan']:
    cur = df[df['Region']==r]
    
    sub.iloc[i*12,-2] = (float(cur[cur['Start Date']=='2018/1/1']['Value'])+float(cur[cur['Start Date']=='2017/1/1']['Value']))/2   
    for j in range(1,12):
        sub.iloc[i*12+j,-2] = sub.iloc[i*12+j-1,-2] * d[r][j-1]
    
    i += 1
    


        
        
        


    
    
#sub.to_csv('./solution.csv')











