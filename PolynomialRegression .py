# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:32:25 2023

@author: Baran Celal Tonyalı
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




 #Veri Yükleme(Data İmport)
 
veriler = pd.read_csv('maaslar.csv')
print(veriler)

#Data Frames Slice
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#Numpy array Dönüşümü
X = x.values
Y = y.values

#Linear Regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x.values,y.values,color='red')
plt.plot(x,lin_reg.predict(x.values),color = 'blue')
plt.show()

#polynomial regression
#doğrusal olmayan model oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

#göreslleştirme
plt.scatter(X,Y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()


#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

#verilerin ölçeklendirilmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color='blue')
plt.show()

print(svr_reg.predict(11))
print(svr_reg.predict(6.6))

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)







