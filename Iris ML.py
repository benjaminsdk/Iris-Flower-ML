# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
dataset_path ="https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv"
X = pd.read_csv(dataset_path)
# Drop any missing data
X = X.dropna()


y = X.pop('class')
# Encode the Labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.3,random_state=0) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

clf = SVC(gamma='auto')
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

import joblib

joblib.dump(clf, 'iris.pkl')

clf2 = joblib.load('iris.pkl')
import numpy as np
X_new = np.array([[6.7,3.1,4.7,1.5]])
y = clf2.predict(X_new)
label = {0:'sentosa',1:'versicolor',2:'virĀinica'}
print('The flower is ',label[y[0]])