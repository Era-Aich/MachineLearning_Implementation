# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:40:21 2025

@author: asus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("D:\Machine_Learning _algorithm_implementation\heart-disease.csv")

x= data.drop("target", axis=1)
y = data["target"]

from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

for i in range(1,100,10):
    
    clf = RandomForestClassifier(n_estimators=i).fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    clf.score(x_test,y_test)
    print(f"Classification score for {i} estimator is : {accuracy_score(y_test,y_pred)}")

