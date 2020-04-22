#  -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:37:03 2020

@author: shivamjadhav
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('data.csv') 
#data Preprocessing
d={0:"Extremely Weak",
   1:"Weak",
   2:"Normal",
   3:"Overweight",
   4:"Obesity",
   5:"Extreme Obesity"}
X=dataset.iloc[:,0:3]
Y=dataset.iloc[:,3]

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X['Gender'] = labelencoder_X.fit_transform(X['Gender'])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
"""
Index : 
0 - Extremely Weak 
1 - Weak 
2 - Normal 
3 - Overweight 
4 - Obesity 
5 - Extreme Obesity
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 100, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
