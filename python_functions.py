# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:04:19 2020

@author: Rajkumar
"""

import pandas as pd
import numpy as np
import random

df = pd.read_excel('dataset/Data_Train.xlsx')
df = pd.read_csv('dataset/creditcard.csv')

''' Setting and re-setting index '''
df = df.reset_index()
df = df.reset_index(drop=True)
df = df.set_index('RATING')

df = df.replace({'RATING':['NEW','-']},'0')
df['RATING'] = df['RATING'].astype('float')

df = df.dropna() # drop rows having NA from dataframe
df1 = df.drop('Row ID',axis=1,inplace=True) # drop one column from dataframe
df = df.fillna('')  # fill na with '' in dataframe

''' Dealing with datetime '''
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['months'] = df['Timestamp'].dt.month_name()
df['day'] = df['Timestamp'].dt.day_name()

''' join 2 dataframes '''
# how: {‘left’, ‘right’, ‘outer’, ‘inner’}
resultant = df.merge(df1, left_on="company_id", right_on="company_id", how='outer')
resultant = df.join(df1, how='left')

''' Concatenation '''
df_cat1 = pd.concat([df1, df2, df3], axis=0)
df_cat1 = pd.concat([df1, df2, df3], axis=1)

''' sortng '''
track = df.sort_values(by=['COST'], ascending=False).reset_index(drop=True)
track = df.sort_values(by=['COST','TITLE'], ascending=False).reset_index(drop=True)

''' Drop duplicate rows '''
df = df.drop_duplicates(keep='first')
drop = df.drop_duplicates(subset=['TITLE'], keep='first')

''' find duplicates rows '''
duplicates = df[df.duplicated(subset=['TITLE'], keep='first')]
duplicates = df[df.duplicated()]

cumsum = df['COST'].cumsum()  ## cumulative sum of crows
x = list(pd.unique(df['TITLE']))  ## find district values
x = list(df['TITLE'].unique())   ## find district values

''' groupby sum, mean, count '''
avg = df.groupby('TITLE').mean()
avg = df.groupby('TITLE')['COST'].mean()
summ = df.groupby('TITLE').sum()
minn = df.groupby('TITLE').min()
count = df.groupby('TITLE').count()
count = df.groupby(['TITLE','CUISINES']).count().reset_index()
count = df['TITLE'].value_counts()

''' second highest cost '''
second = df.sort_values(by='COST', ascending=False).iloc[1,:]

''' pivot table '''
table = pd.pivot_table(df, index=["TITLE"], aggfunc=np.sum, fill_value=0)
table = df.pivot_table(values=['Sales','Quantity','Profit'],index=['Region','State'], aggfunc='mean')

''' apply method '''
table = df['Customer Name Length'] = df['Customer Name'].apply(len)
table = df['Discounted Price'] = df['Sales'].apply(lambda x:0.85*x if x>200 else x)

''' datatypes conversion '''
x = [1,2,3,4,5,2]
m = np.array([1,2])
x.count(3)
x[::-1]
y=tuple(x)
z=list(y)
z=set(x)
z.add('b')
y=z.copy()

''' single line for loop with ifelse '''
y = ['a' if i==2 else i for i in x]

''' find intersection b/w 2 sets '''
z.difference(set(x))

''' generate random int within range '''
random.randint(1,7)

'''sum of number'''
def sum_num(n):
    n = str(n)
    a = [int(i) for i in list(n)]
    b = sum(a)
    return b
# sum_num(123) == 6

''' reverse the number '''
def reverse_num(n):
    n = str(n)
    a = [i for i in list(n)]
    a = a[::-1]
    b = ''.join(a)
    b = int(b)
    return b
# reverse_num(123) == 321

''' check palindrome number'''
def check_palindrome(n):
    n = str(n)
    a = [i for i in list(n)]
    b = a[::-1]
    if a == b:
        output = 'palindrome'
    else:
        output = 'No'
    return output
# check_palindrome(123321) == 'palindrome'

'''count the digit in number '''
def count_digit(n):
    n = str(n)
    a = [i for i in list(n)]
    b = len(a)
    return b
# count_digit(1234) == 4

''' find second largest number in list '''
def second_large(a):
    a.sort()
    b= a[-2]
    return b
# second_large([1,2,3,4]) == 3

''' check prime number '''
def check_prime(n):
    for i in range(2,n):
        print(str(n%i))
        if n%i == 0:
            return 'No'
        else:
            pass
    return 'Prime'
# check_prime(9)












