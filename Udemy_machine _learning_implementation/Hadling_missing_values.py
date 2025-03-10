# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:02:45 2025

@author: asus
"""

import numpy as np
import pandas as pd

data = pd.read_csv("D:\Machine_Learning _algorithm_implementation\car-sales-extended-missing-data.csv")

print(data.isna().sum())

data.dropna(subset=["Price"],inplace=True)

print(data.isna().sum())

x= data.drop("Price",axis=1)
y=data["Price"]

##Handling missing values

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


cat_impute = SimpleImputer(strategy="constant",fill_value="missing")
num_impute = SimpleImputer(strategy="mean")
door_impute = SimpleImputer(strategy="constant",fill_value=4)

cat_imputes = ["Make","Colour"]
num_imputes = ["Odometer (KM)"]
door_imputes = ["Doors"]

car_transformer = ColumnTransformer([("cat_impute",cat_impute,cat_imputes),("num_impute",num_impute,num_imputes),("door_impute",door_impute,door_imputes)])

x_transform = car_transformer.fit_transform(x)
print(x_transform)

dataframe = pd.DataFrame(x_transform,columns=["Make","Colour","Odometer (KM)","Doors"])



## Convert Categorail value to numerical values

from sklearn.preprocessing import OneHotEncoder

onehot= OneHotEncoder()

Categorail=["Make","Colour"]

coded_features = ColumnTransformer([("onehot",onehot,Categorail)], remainder="passthrough")

x_coded = coded_features.fit_transform(dataframe)


## Import Model
np.random.seed(42)

from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_coded,y,test_size=0.2)


## Fit data in the model 

clf.fit(x_train,y_train)

print(clf.score(x_test, y_test))


##Import accuracy metrics

from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = clf.predict(x_test)





