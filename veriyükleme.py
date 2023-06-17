# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:45:39 2023

@author: Baran Celal Tonyalı
"""

#Kütüphane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Kodlar
 #Veri Yükleme(Data İmport)
 
veriler = pd.read_csv('veriler.csv')
print(veriler)

 #Veri Ön İşleme
 
boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

#Class tanımlama ve düzenleme

class insan: 
    boy = 180
    def kosmak (self,b) 
    
    
ali = insan()        
print(ali.boy)
print(ali.kosmak(90))

#liste tanımlama
l = [1,3,4]

#Eksik Veriler



    