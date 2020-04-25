# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:32:19 2020

@author: Rajkumar
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.svm import SVC, SVR
from sklearn.externals import joblib

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

import Data_preprocessing
df = pd.read_excel('dataset/Data_Train.xlsx')

Data_preprocessing.summary(df)
cleaned_df = Data_preprocessing.datacleaning(df)
cleaned_df = Data_preprocessing.treat_missingValue(cleaned_df)
cleaned_df = Data_preprocessing.treat_outliers(cleaned_df)
feature = Data_preprocessing.featureEngineering(cleaned_df)
selected_features = Data_preprocessing.Feature_selection(feature, num_features, target_col)

''' Split the training and tesing datasets from main datasets '''
X,y = selected_features[cols], selected_features['COST']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)

'''                                   1. train the support vector classifier model                                '''
model = SVC()
model.fit(X_train,y_train)

'''Predict the result '''
predictions = model.predict(X_test)

'''Evaluation of classifier '''
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

'''
there are many optimization methods such as: 
1. Grid search 2. random search 3. Bayesian optimization
'''
def Gridsearch_SVM(model):
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
						{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
## gridsearchCV & cross_validate
    grid = GridSearchCV(model, parameters, cv=5)
    grid_result = grid.fit(X_train, y_train)
# summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid

'''run gridsearch function to find the optimum value of model parameters'''
grid = Gridsearch_SVM(SVC())
predictions = grid.predict(X_test)

'''Evaluation of classifier '''
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

'''                                  2. train the support vector regression model                               '''
model = SVR()
model.fit(X_train,y_train)

'''Predict the result '''
predictions = model.predict(X_test)

'''Evaluation of regression '''
print(model.score(X_test,y_test))

def Gridsearch_SVM(model):
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'epsilon':[0.1,0.2,0.5,1],'C': [1, 10, 100, 1000]},
						{'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'epsilon':[0.1,0.2,0.5,1]}]
## gridsearchCV & cross_validate
    grid = GridSearchCV(model, parameters, cv=5)
    grid_result = grid.fit(X_train, y_train)
# summarize results
    print("Score: "+ str(grid_result.score(X_test,y_test)))
    print("RMSE for model: "+ str(np.sqrt(mean_squared_error(y_test, grid_result.predict(X_test)))))
    print("Best estimator: "+ str(grid.best_estimator_))
    return grid

'''run gridsearch function to find the optimum value of model parameters'''
grid = Gridsearch_SVM(SVC())
predictions = grid.predict(X_test)