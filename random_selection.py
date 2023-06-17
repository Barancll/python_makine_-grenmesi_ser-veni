# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:31:21 2023

@author: Baran Celal Tonyalı
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

#Random Selection
import random 

N = 10000
d = 10
toplam = 0
secilenler = []
for n in range (0,N):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] #verilerdeki n. satır = 1 ise odul 1
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()

#UCB 


   