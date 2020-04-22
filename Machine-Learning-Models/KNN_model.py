# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:59:56 2020

@author: Rajkumar
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

import Data_preprocessing
df = pd.read_excel('dataset/Data_Train.xlsx')

Data_preprocessing.summary(df)
cleaned_df = Data_preprocessing.datacleaning(df)
cleaned_df = Data_preprocessing.treat_missingValue(cleaned_df)
cleaned_df = Data_preprocessing.treat_outliers(cleaned_df)
feature = Data_preprocessing.featureEngineering(cleaned_df)
selected_features = Data_preprocessing.Feature_selection(feature)

''' Split the training and tesing datasets from main datasets '''
X,y = selected_features[cols], selected_features['COST']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)

''' train the KNN model '''
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

'''Predict the result '''
pred = knn.predict(X_test)

'''Evaluation of classification quality '''
conf_mat=confusion_matrix(y_test,pred)
print(conf_mat)
print(classification_report(y_test,pred))
print("Misclassification error rate:",round(np.mean(pred!=y_test),3))

'''Choosing 'k' by elbow method'''
error_rate = []

for i in range(1,60):    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value', fontsize=20)
plt.xlabel('K',fontsize=15)
plt.ylabel('Error (misclassification) Rate',fontsize=15)
