# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:04:19 2020

@author: Rajkumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('dataset/Data_Train.xlsx')
df = pd.read_csv('dataset/creditcard.csv')

df = df.replace({'RATING':['NEW','-']},'0')
df['RATING'] = df['RATING'].astype('float')
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
    return df_cleaned
df = df.dropna()
df = df.fillna('')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['months'] = df['Timestamp'].dt.month_name()
df['day'] = df['Timestamp'].dt.day_name()
## join 
resultant = df.merge(company_df, left_on="company_id", right_on="company_id",how='outer')
# how: {‘left’, ‘right’, ‘outer’, ‘inner’}
## sortng
track = df.sort_values(by=['COST'], ascending=False).reset_index(drop=True)
track = df.sort_values(by=['COST','TITLE'], ascending=False).reset_index(drop=True)
# find district values
x = list(pd.unique(track['TITLE']))
## duplicates
df = df.drop_duplicates(keep='first')
drop = df.drop_duplicates(subset=['TITLE'], keep='first')
# find duplicates rows
duplicates = df[df.duplicated(subset=['TITLE'], keep='first')]
duplicates = df[df.duplicated()]
## cumulative sum of crows
cumsum = df['COST'].cumsum()

## groupby sum,mean, count
avg = df.groupby('TITLE').mean()
avg = df.groupby('TITLE')['COST'].mean()
summ = df.groupby('TITLE').sum()
minn = df.groupby('TITLE').min()
count = df.groupby('TITLE').count()
count = df.groupby(['TITLE','CUISINES']).count().reset_index()
count = df['TITLE'].value_counts()

## second highest cost
second = df.sort_values(by='COST', ascending=False).iloc[1,:]
## pivot table
table = pd.pivot_table(df, index=["TITLE"], aggfunc=np.sum, fill_value=0)

# Exploratory Data Analysis ###
f, axes = plt.subplots(2, 2)
sns.boxplot(x=df['COST'], ax=axes[0, 0])
sns.boxplot(x=df['VOTES'], ax=axes[0,1])
sns.boxplot(x=df['RATING'] , ax=axes[1,0])
sns.pairplot(df,vars=['COST','RATING'], kind='scatter')
sns.countplot(y=df["CITY"])
sns.countplot(y=df["CUISINES"])
sns.lineplot(x='months',y='COST',data=df, estimator=np.median)
sns.barplot(x="COST", y="CITY", data=df, estimator=np.median)
table=pd.crosstab(df["CUISINES"], df['TITLE'])
table.plot(kind='barh',stacked=True)

import numpy as np
import random
x = [1,2,3,4,5,2]
y=tuple(x)
z=list(y)
z=set(x)
x.count(3)
['a' if i==2 else i for i in x]
np.array([1,2])
x[::-1]
y=z.add('a')

y=z.copy()
z.difference(set(x))
random.randint(1,7)
x*2
def summ(n):
    n = str(n)
    a = [int(i) for i in list(n)]
    b = sum(a)
    return b

def rev(n):
    n = str(n)
    a = [i for i in list(n)]
    a = a[::-1]
    b = ''.join(a)
    b = int(b)
    return b

def check(n):
    n = str(n)
    a = [i for i in list(n)]
    b = a[::-1]
    if a == b:
        output = 'palindrome'
    else:
        output = 'No'
    return output

def count(n):
    n = str(n)
    a = [i for i in list(n)]
    b = len(a)
    return b

def second_large(a):
    a.sort()
    b= a[-2]
    return b

def swap(a):
    temp = a[0]
    a[0] = a[-1]
    a[-1] = temp
    return a

def check_prime(n):
    for i in range(2,n):
        if n%i == 0:
            return 'No'
        else:
            return 'Prime'













