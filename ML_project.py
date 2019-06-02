# -*- coding: utf-8 -*-
"""
@author: Rajkumar
"""
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score
import os
os.chdir(os.path.dirname(__file__))
# Read Excel file #####
df = pd.read_excel('Participants_Data_Final\Data_Train.xlsx')

# data cleaning #####

# missing values treatment ###

# outlier detection & treatment ###

# feature engineering ###

# EDA ###

# feature selection ###

## Split the training and tesing datasets from main datasets
X,y = df.iloc[:,1:11], df.iloc[:,11]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


## import support vector machine #####
from sklearn.svm import SVC
model = SVC()


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
    return None

# check the accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


