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

# data summarization ###
def summary(df):
	df.describe()
	df.describe(include=['object'])
	df.info()
	df.head(5)
	return None

# data cleaning #####
def datacleaning(df, special_character1=False, digit=False,
                 nalpha=False):
	df = df.drop_duplicates(keep='first')
	df = df.reset_index(drop=True)
# for title case
	df = df.apply(lambda x: x.str.strip().str.title() if(x.dtypes == object)
				  else x)
    if special_character1:
        df = df.apply(lambda x: x.str.split(r'[^a-zA-Z.\d\s]').
                      str[0] if(x.dtypes == object) else x)
    if digit:
        df = df.apply(lambda x: x.str.replace(r'\d+', '')
                      if(x.dtypes == object) else x)
    if nalpha:
        df = df.apply(lambda x: x.str.replace(r'\W+', ' ')
                      if(x.dtypes == object) else x)
	df = df.apply(lambda x: x.str.strip().str.strip() if(x.dtypes == object)
				  else x)
	df['Route'] = df['Route'].str.replace('New Delhi-Cochin', 'Delhi-Cochin')	
	return df

# missing values treatment ###
def treat_missingValue(df):
	df = df.apply(lambda x: x.fillna(x.median(), inplace=True)
				  if((x.dtypes != object) & (x.isnull().sum() > 0)) else x)
	return df

# outlier detection & treatment ###
def detect_outliers(col):
    outlier1 = []
    threshold = 3
    mean = np.mean(col)
    std =np.std(col)
    for i in col:
        z_score = (i - mean)/std 
        if np.abs(z_score) > threshold:
            outlier1.append(i)
    outlier2 = []
    sorted(col)
    q1, q3 = np.percentile(col,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr)
    for i in col:
        if ((i > upper_bound) | (i < lower_bound)):
            outlier2.append(i)
    lst1 = np.unique(outlier1)
    lst2 = np.unique(outlier2)
    output = list(set(lst1) & set(lst2))
    return output 

# treat outliers ###
def treat_outliers(df):
#	x = treat_outliers(df.Duration)
#	y = df.Duration.replace(x, df.Duration.median())
	df = df.apply(lambda x: x.replace(treat_outliers(x), x.median())
				  if((x.dtypes != object) & (len(treat_outliers(x)) > 0)) else x)
	return df

# feature engineering ###
def featureEngineering(df):
	df['time'] = pd.to_datetime(df['time'])
	df['months'] = df['time'].dt.month_name()
	df['day'] = df['time'].dt.day_name()
	df = df.sort_values(by='time')
	lb_make = LabelEncoder()
	y_test = lb_make.fit_transform(df.Holiday)
	return df

# Exploratory Data Analysis ###
f, axes = plt.subplots(2, 2)
sns.boxplot(x=df["Dep_Time"], ax=axes[0, 0])
sns.boxplot(x=df["Arrival_Time"], ax=axes[0,1])
sns.boxplot(x=df["Duration"] , ax=axes[1,0])
sns.boxplot(x=df["Price"], ax=axes[1,1])
sns.pairplot(tab,vars=['Price','Duration','Arrival_Time','Dep_Time'], kind='scatter')
f, axes = plt.subplots(1, 2)
sns.countplot(y=df["Airline"], ax=axes[0])
sns.countplot(y=df["Route"], ax=axes[1])

# Feature Selection ###
def Feature_selection(df):
	return df

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


