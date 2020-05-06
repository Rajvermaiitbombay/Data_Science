# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:27:02 2020

@author: Rajkumar
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestRegressor

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)


''' data summarization '''
def summary(df):
    print(df.shape)
    print('--- Description of numerical variables')
    print(df.describe())
    print('--- Description of categorical variables')
#    print(df.describe(include=['object']))
    print('--- Gerenal information about variables')
    print(df.info())
    print('--- view the 5 rows of dataset')
    print(df.head(5))
    return None

''' data cleaning '''
def datacleaning(df, special_character1=False, digit=False,
                 nalpha=False):
    df = df.drop_duplicates(keep='first')
    df = df.reset_index(drop=True)
    df_cleaned = df.copy()
    if special_character1:
        df_cleaned = df_cleaned.apply(lambda x: x.str.split(r'[^a-zA-Z.\d\s]').
                                      str[0] if(x.dtypes == object) else x)
    if digit:
        df_cleaned = df_cleaned.apply(lambda x: x.str.replace(r'\d+', '')
                                      if(x.dtypes == object) else x)
    if nalpha:
        df_cleaned = df_cleaned.apply(lambda x: x.str.replace(r'\W+', ' ')
                                      if(x.dtypes == object) else x)
    df_cleaned = df_cleaned.apply(lambda x: x.str.strip() if(x.dtypes == object)
                                  else x)
    df_cleaned['CITY'] = df_cleaned['CITY'].str.replace('Kochi', 'Cochin')
    replace_values = {'kanpur':['kanpur nagar','kanpur dehat']}
    df_cleaned['CITY'] = df_cleaned['CITY'].replace(replace_values['kanpur'], 'kanpur')
    return df_cleaned

''' missing values treatment '''
def treat_missingValue(df):
    # for numerical columns
    df = df.apply(lambda x: x.fillna(x.median())
				  if(x.dtypes != object) else x)
    # for categorical columns
    df = df.dropna()
    return df


''' outlier detection & treatment '''
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

''' treat outliers '''
def treat_outliers(df):
	df = df.apply(lambda x: x.replace(detect_outliers(x), x.median())
				  if(x.dtypes != object) else x)
	return df

''' Get categorical columns '''
def get_categoricals(df):    
    categoricals = []
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'O':
             categoricals.append(col)
        else:
             df[col].fillna(0, inplace=True)
    return categoricals


''' feature engineering '''
def featureEngineering(df):
    vote = df.VOTES.str.split(' ', n = 1, expand = True)
    df['VOTES'] = vote[0]
    df['VOTES'] = df['VOTES'].astype('int64')
    df['RATING'] = df['RATING'].astype('float64')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['months'] = df['Timestamp'].dt.month_name()
    df['day'] = df['Timestamp'].dt.day_name()
    df = df.sort_values(by='Timestamp')
    # Label Encoding
    lb_make = LabelEncoder()
    df.CITY = lb_make.fit_transform(df.CITY)
    # One hot Encoding
    df = pd.get_dummies(df, columns=get_categoricals(df), dummy_na=True)
    return df

''' Feature Selection '''
def Feature_selection(df, num_features, target_col):
    df = df.apply(lambda x: x.astype('float64'))
    #Backward Elimination with R-squared
    X = df.drop(target_col, axis=1)
    y = df[[target_col]]
    backwardModel = sfs(RandomForestRegressor(),k_features= num_features,forward=False,cv=5,n_jobs=-1,scoring='r2')
#    scoring = 'roc-auc' used for classification
    backwardModel.fit(np.array(X),y)
    features1 = list(X.columns[list(backwardModel.k_feature_idx_)])
    print('selected features set1:' + str(features1))
    
    # using statistics method
    model = SelectKBest(chi2, k=num_features)
    fit = model.fit(X, y)
    features = fit.transform(X)
    selected = features[0:1,:].tolist()[0]
    cols = X.iloc[0:1,:].values.tolist()[0]
    ind_dict = dict((k,i) for i,k in enumerate(cols))
    inter = set(ind_dict).intersection(selected)
    indices = [ ind_dict[x] for x in inter]
    features2 = [list(X.columns)[i] for i in indices]
    print('Selected features set2:' + str(features2))
    
    # Recursive Feature Elimination
    model = LogisticRegression()
    rfe = RFE(model, num_features)
    fit = rfe.fit(X, y)
    features = fit.transform(X)
    cols = list(X.columns)
    selected = list(fit.support_)
    features = [i if j==True else '' for i,j in zip(cols, selected)]
    features3 = list(pd.unique(features))
    features3.remove('')
    print("Feature Ranking: %s" % (fit.ranking_))
    print("Selected Features set3: %s" % (features3))
    features = list(set(features1).intersection(features2).intersection(features3))
    return features


if __name__ == '__main__':
    ''' read dataset '''
    df1 = pd.read_csv('dataset/winequality-red.csv')
    cols = list(df1.columns)[0].split(';')
    cols = [i.replace(r'"', '') for i in cols]
    df1.columns = ['col']
    df1 = pd.concat([df1,df1.col.str.split(';',expand=True)],1)
    df2 = pd.read_csv('dataset/winequality-white.csv')
    df2.columns = ['col']
    df2 = pd.concat([df2,df2.col.str.split(';',expand=True)],1)
    frames = [df1, df2]
    df = pd.concat(frames)
    df = df.reset_index(drop=True)
    df = df.drop('col', axis=1)
    df.columns = cols
    ''' preprocessing data '''
    summary(df)
    cleaned_df = treat_missingValue(df)
    cleaned_df = treat_outliers(cleaned_df)
    selected_features = Feature_selection(cleaned_df, 10, 'quality')
