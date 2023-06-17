# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:12:46 2023

@author: bct
"""

import pandas as pd

#Dosyayı kaydetmeden url üzerinden açma
url = "http://bilkav.com/satislar.csv"

veriler = pd.read_csv(url)
veriler = veriler.values

X = veriler[:,0:1]
Y = veriler[:,1]

bolme = 0.33

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = bolme)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
print(lr.predict(X_test))


#pickle ile kaydetme 
import pickle
dosya = "model.kayıt"
pickle.dump(lr,open(dosya,'wb'))


#pickle ile yüklenen dosyayı açma
yuklenen = pickle.load(open(dosya,'rb'))
print(yuklenen.predict(X_test))
