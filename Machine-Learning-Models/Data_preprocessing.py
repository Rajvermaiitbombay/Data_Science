# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:27:02 2020

@author: Rajkumar
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)


''' data summarization '''
def summary(df):
    print(df.shape)
    print('--- Description of numerical variables')
    print(df.describe())
    print('--- Description of categorical variables')
    print(df.describe(include=['object']))
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
    df_cleaned['CITY'] = df_cleaned['CITY'].str.replace('New Delhi', 'Delhi')
    df_cleaned['CITY'] = df_cleaned['CITY'].str.replace('Delhi NCR', 'Delhi')
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
def Feature_selection(df):
    #Backward Elimination
    X = df.iloc[:,1:11]
    y = df.iloc[:,11]
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y,X_1).fit()
        p = pd.Series(model.pvalues.values[1:],index = cols)      
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    print('selected_features_set1:' + str(cols))
    
    # using statistics method
    X_new = SelectKBest(chi2, k=6).fit_transform(X, y)
    cols = X_new.columns
    print('selected_features_set2:' + str(cols))
    
    # Recursive Feature Elimination
    model = LogisticRegression()
    rfe = RFE(model, 6)
    fit = rfe.fit(X, y)
    print("Feature Ranking: %s" % (fit.ranking_))
    return None

''' Exploratory Data Analysis '''
def EDA(feature):
    f, axes = plt.subplots(2, 2)
    sns.boxplot(x=feature['COST'], ax=axes[0, 0])
    sns.boxplot(x=feature['VOTES'], ax=axes[0,1])
    sns.boxplot(x=feature['RATING'] , ax=axes[1,0])
    sns.pairplot(feature,vars=['COST','VOTES','RATING'], kind='scatter')
    sns.countplot(y=feature["CITY"])
    sns.countplot(y=feature["CUISINES"])
    sns.countplot(x='CUISINES',hue='months',data=feature, palette='Set1')
    sns.lineplot(x='months',y='COST',data=feature, estimator=np.median)
    sns.barplot(x="COST", y="CITY", data=feature, estimator=np.median)
    table=pd.crosstab(feature["CUISINES"], feature['CITY'])
    table.plot(kind='barh',stacked=True)
    return None

if __name__ == '__main__':
    df = pd.read_excel('dataset/Data_Train.xlsx')
    summary(df)
    cleaned_df = datacleaning(df)
    cleaned_df = treat_missingValue(cleaned_df)
    cleaned_df = treat_outliers(cleaned_df)
    feature = featureEngineering(cleaned_df)
    selected_features = Feature_selection(feature)