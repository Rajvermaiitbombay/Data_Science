# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:26:07 2019

@author: Rajkumar
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_excel('data.xlsx', encoding='latin1')

df = df[['LSP Name','Client','Sector','HQ','TruckType']]
df.head()

df = df.apply(lambda x: x.str.strip().str.title() if(x.dtypes == object)
              else x)

def data():
    dfh = pd.DataFrame({'LSP Name':[], 'Client':[], 'Sector':[], 'HQ':[], 'TruckType':[]})
    for i in list(pd.unique(df['LSP Name'])):
        df1 = df[df['LSP Name']==i].reset_index(drop=True)
        a = []
        b = []
        c = []
        for ind in range(len(df1)):
            a.append(df1['Client'][ind])
            b.append(df1['Sector'][ind])
            c.append(df1['TruckType'][ind])
        a=list(pd.unique(a))
        b=list(pd.unique(b))
        c=list(pd.unique(c))
        dummy = pd.DataFrame({'LSP Name':[i], 'Client':[' '.join(a)], 'Sector':[' '.join(b)],
                              'HQ':[df1['HQ'][0]], 'TruckType':[' '.join(c)]})
        dfh = dfh.append(dummy)
    dfh = dfh.reset_index(drop=True)
    return dfh        

dfh = data()

def create_soup(x):
    return ''.join(x['Client']) + ' ' + ''.join(x['Sector'])+ ''.join(x['HQ'])+ ''.join(x['TruckType'])

dfh['para'] = dfh.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')

#Construct the required count matrix by fitting and transforming the data
count_matrix = count.fit_transform(dfh['para'])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use in the function to match the indexes
indices = pd.Series(dfh.index, index=dfh['LSP Name'])

#  defining the function that takes in movie title 
# as input and returns the top 10 recommended movies
def recommendations(lsp, cosine_sim = cosine_sim):
    
    # initializing the empty list of recommended movies
    recommended_movies = []
    
    # gettin the index of the movie that matches the title
    idx = indices[lsp]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(dfh['LSP Name'])[i])
    print(top_10_indexes)    
    return recommended_movies