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




sub = pd.read_csv('./solution.csv')

i = 0
for r in ['Alberta','British Columbia', 'Manitoba', 'Ontario','Québec','Saskatchewan']:
    cur = df[df['Region']==r]

    for j in range(12):
        sub.iloc[i*12+j,-2] = (float(cur[cur['Start Date']=='2018/'+str(j+1)+'/1']['Value'])+\
                float(cur[cur['Start Date']=='2017/'+str(j+1)+'/1']['Value']))/2   
        
        if sub.iloc[i*12+j,-2] == 0:
            sub.iloc[i*12+j,-2] = 700
    
    i += 1
    


        


    
    
sub.to_csv('./solution.csv')











