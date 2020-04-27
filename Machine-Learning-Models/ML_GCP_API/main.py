# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:06:49 2020

@author: Rajkumar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:32:19 2020

@author: Rajkumar
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.externals import joblib
from bson.json_util import dumps

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

import Data_preprocessing
import blob
directory = '/tmp/'
blob_url = 'https://logisticsnowtech3.blob.core.windows.net/rajkumar/'

def data_preparation(feature_selection=False):
    ''' read dataset '''
    df1 = pd.read_csv(blob_url + 'winequality-red.csv')
    cols = list(df1.columns)[0].split(';')
    cols = [i.replace(r'"', '') for i in cols]
    df1.columns = ['col']
    df1 = pd.concat([df1,df1.col.str.split(';',expand=True)],1)
    df2 = pd.read_csv(blob_url + 'winequality-white.csv')
    df2.columns = ['col']
    df2 = pd.concat([df2,df2.col.str.split(';',expand=True)],1)
    frames = [df1, df2]
    df = pd.concat(frames)
    df = df.reset_index(drop=True)
    df = df.drop('col', axis=1)
    df.columns = cols
    df = df.apply(lambda x: x.astype('float64'))
    Data_preprocessing.summary(df)
    cleaned_df = Data_preprocessing.treat_missingValue(df)
    cleaned_df = Data_preprocessing.treat_outliers(cleaned_df)
    if feature_selection:
        features = Data_preprocessing.Feature_selection(cleaned_df, 10, 'quality')
    else:
        features = ['volatile acidity', 'sulphates', 'citric acid', 'alcohol', 'pH',
                   'residual sugar', 'free sulfur dioxide', 'chlorides']
    selected_features = cleaned_df[features]
    target = cleaned_df['quality']
    return selected_features, target

''' Split the training and tesing datasets from main datasets '''

def data_segregation(selected_features, target):
    X_train, X_test, y_train, y_test = train_test_split(selected_features, target,test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

''' there are many optimization method '''

def Gridsearch_SVM(model, X_train, X_test, y_train, y_test):
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

''' train the support vector regression model '''

def model_training(selected_features, target):
    X_train, X_test, y_train, y_test = data_segregation(selected_features, target)
    model = SVR()
    model.fit(X_train,y_train)
#    model = Gridsearch_SVM(SVC(), X_train, X_test, y_train, y_test)
    return model

'''Export the ML model  '''
def export_model(model):
    joblib.dump(model, directory+'mymodel.pkl')
    blob.writeBlob(directory+'mymodel.pkl', 'mymodel.pkl', 'rajkumar')
    return None

def export_columns(selected_features):
    model_cols = list(selected_features.columns)
    joblib.dump(model_cols, directory+'model_columns.pkl')
    blob.writeBlob(directory+'model_columns.pkl', 'model_columns.pkl', 'rajkumar')
    return None

'''Predict the result '''
def prediction(test_data):
    model = blob.getBlob_stream('mymodel.pkl', container_name='rajkumar') # Load "model.pkl"
    print ('Model loaded')
    model_columns = blob.getBlob_stream("model_columns.pkl", container_name='rajkumar') # Load "model_columns.pkl"
    print ('Model columns loaded')
    test_data = test_data.reindex(columns=model_columns, fill_value=0)
    output = model.predict(test_data)
    return output    

'''train existing model '''

def train_model(key='repeat'):
    try:
        selected_features, target = data_preparation(feature_selection=False)
        X_train, X_test, y_train, y_test = data_segregation(selected_features, target)
        model = model_training(selected_features, target)
        if key == 'initial':
            export_model(model)
            export_columns(selected_features)
        else:
            score1 = model.score(X_test, y_test)
            existing_model = blob.getBlob_stream('mymodel.pkl', container_name='rajkumar') # Load "model.pkl"   
            score2 = existing_model.score(X_test, y_test)
            if score1 > score2:
                export_model(model)
            else:
                return 'existing model is good'
        return 'successfully updated'
    except Exception:
        return 'fail'

'''Evaluation of regression '''
def evaluate_model():
    selected_features, target = data_preparation(feature_selection=False)
    X_train, X_test, y_train, y_test = data_segregation(selected_features, target)
    model = blob.getBlob_stream('mymodel.pkl', container_name='rajkumar') # Load "model.pkl"   
    score = model.score(X_test, y_test)
    return score

def main_func(request):
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for main requests
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    req_body = request.get_json()
    if req_body is None:
        error = 'test data not found'
        return (f"{error}!", 401, headers)
    else:
        purpose = req_body['purpose']
    if purpose == 'prediciton':
        try:
            data = req_body['data']
            test_data = pd.DataFrame(data)
            output = prediction(test_data)
            dictt = {'output': output}
            return (dumps(dictt), 200, headers)
        except Exception as e:
            dictt = {"error": str(e)}
            return (dumps(dictt), 401, headers)
    elif purpose == 'training':
        key = req_body['key']
        status = train_model(key)
        dictt = {'status': status}
        return (dumps(dictt), 200, headers)
    elif purpose == 'evaluation':
        try:
            score = evaluate_model()
            dictt = {'score': score}
            return (dumps(dictt), 200, headers)
        except Exception as e:
            dictt = {"error": str(e)}
            return (dumps(dictt), 401, headers)        
        
    
    