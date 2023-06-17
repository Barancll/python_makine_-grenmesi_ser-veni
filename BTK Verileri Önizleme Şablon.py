# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:45:39 2023

@author: Baran Celal Tonyalı
"""

#1.Kütüphane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.Verileri Önişleme
 #2.1.Veri Yükleme(Data İmport)
 
veriler = pd.read_csv('veriler.csv')
print(veriler)

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

 #2.6.Verileri Eğitim Ve Test İşlemi İçin Bölme
 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

 #2.7.Öznitelik Ölçekleme

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#LogisticRegression Uygulaması
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0) 
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


#K-NN Algoritmaları 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print (cm)

#SVM Algoritması
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)


