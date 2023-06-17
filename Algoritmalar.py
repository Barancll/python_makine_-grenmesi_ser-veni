# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:45:39 2023

@author: Baran Celal Tonyalı
"""

#1.Kütüphane
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Verileri Önişleme
 #Veri Yükleme(Data İmport)
 
veriler = pd.read_csv('veriler.csv')
print(veriler)

x = veriler.iloc[:,1:4].values #Bağımsız Değişken
y = veriler.iloc[:,4:].values #Bağımlı Değişken

 #Verileri Eğitim Ve Test İşlemi İçin Bölme
 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

 #Verilerin Ölçeklemesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#LogisticRegression Uygulaması
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0) 
logr.fit(X_train,y_train)#Eğitim

y_pred = logr.predict(X_test)#Tahmin
print(y_pred)
print(y_test)

#Confusion Matrix/Karmaşıklık Matrisi
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

#SVC (SVM) Algoritması
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)


#Naive Bayes Algoritması
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

#Decition Tree Algoritması
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


#Random Forest Algoritması
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


#ROC, TPR, FPR Değerleri
from sklearn import metrics
y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)







