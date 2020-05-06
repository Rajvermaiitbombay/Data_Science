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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.externals import joblib
from bson.json_util import dumps
import azure.functions as func
import logging
import openpyxl
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import json


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
    target = pd.get_dummies(target).values
    selected_features = np.asanyarray(selected_features)
    target = np.asanyarray(target)
    return selected_features, target

''' Split the training and tesing datasets from main datasets '''

def data_segregation(selected_features, target):
    X_train, X_test, y_train, y_test = train_test_split(selected_features, target,test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def error_rate(p, t):
  return np.mean(p != t)

''' Build Neural network model '''

def build_model(prediction=False):
    global X, Y, hidden_w, output_w, hidden_b, output_b
    selected_features, target = data_preparation(feature_selection=False)
    X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.33, random_state=42)
    num_features = X_train.shape[1]
    num_labels = y_train.shape[1]  
    # Input and Target placeholders 
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.int64, shape=[None, num_labels])

    num_hidden = int((num_features+num_labels)/2)
    seed = 120   
    hidden_w = tf.Variable(tf.random_normal([num_features, num_hidden], seed=seed))
    output_w =  tf.Variable(tf.random_normal([num_hidden, num_labels], seed=seed))
    
    hidden_b = tf.Variable(tf.random_normal([num_hidden], seed=seed))
    output_b = tf.Variable(tf.random_normal([num_labels], seed=seed))
    
    hidden_layer = tf.add(tf.matmul(X, hidden_w), hidden_b)
    
    output_layer = tf.matmul(hidden_layer, output_w) + output_b
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y))
    if prediction:
        predict = tf.nn.softmax(output_layer)
    else:
        predict = tf.argmax(output_layer, 1)
    return cost, predict

def fit_model(Xtrain, Ytrain, Xtest, Ytest, savefile):

    N = Xtrain.shape[0]
    # hyperparams
    max_iter = 30
    lr = 1e-3
    mu = 0.9
    regularization = 1e-1
    batch_sz = 100
    n_batches = N // batch_sz

    cost, predict = build_model(prediction=False)
    l2_penalty = regularization*tf.reduce_mean(hidden_w**2) / 2
    cost += l2_penalty
    train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)

    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, Y: Ybatch})
                if j % 200 == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, Y: Ytest})
                    Ptest = session.run(predict, feed_dict={X: Xtest})
                    err = error_rate(Ptest, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    costs.append(test_cost)

        # save the model
        saver = tf.train.Saver({'hidden_w': hidden_w, 'output_w': output_w, 'hidden_b': hidden_b,
                                'output_b': output_b}) 
        saver.save(session, savefile)

    plt.plot(costs)
    plt.show()
    session.close()


''' train the support vector regression model '''

def model_training():
    savefile = directory+"tf.model"
    selected_features, target = data_preparation(feature_selection=False)
    X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.33, random_state=42)
    fit_model(X_train, y_train, X_test, y_test, savefile)
    return 'pass'

'''Export the ML model  '''
def export_model(filename):
    checkpoint = directory+"checkpoint"
    blob.writeBlob(checkpoint, "checkpoint", 'rajkumar')
    model = directory+"tf.model.data-00000-of-00001"
    blob.writeBlob(model, "tf.model.data-00000-of-00001", 'rajkumar')
    index = directory+"tf.model.index"
    blob.writeBlob(index, "tf.model.index", 'rajkumar')
    meta = directory+"tf.model.meta"
    blob.writeBlob(meta, "tf.model.meta", 'rajkumar')
    return None

'''Predict the result '''
def prediction(test_data, savefile):
    blob.getBlob("checkpoint", directory+"checkpoint", container_name='rajkumar')
    blob.getBlob("tf.model.data-00000-of-00001", directory+"tf.model.data-00000-of-00001", container_name='rajkumar')
    blob.getBlob("tf.model.index", directory+"tf.model.index", container_name='rajkumar')
    blob.getBlob("tf.model.meta", directory+"tf.model.meta", container_name='rajkumar')
    with tf.Session() as session:
        cost, predict = build_model(prediction=True)
        saver = tf.train.Saver({'hidden_w': hidden_w, 'output_w': output_w, 'hidden_b': hidden_b,
                                'output_b': output_b})
        saver.restore(session, savefile)
        output = session.run(predict, feed_dict={X: test_data})
    test_data = pd.DataFrame(test_data)
    test_data['output'] = output
    test_data.to_excel(directory+'result.xlsx')
    blob.writeBlob(directory+'result.xlsx', 'result.xlsx', 'rajkumar')    
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
    
def score(X, Y, savefile):
    return 1 - error_rate(predict(X, savefile), Y)

'''Evaluation of regression '''
def evaluate_model():
    selected_features, target = data_preparation(feature_selection=False)
    X_train, X_test, y_train, y_test = data_segregation(selected_features, target)
    savefile = blob.getBlob_stream("tf.model", container_name='rajkumar')  
    score = model.score(X_test, y_test)
    return score
    

def main(req: func.HttpRequest) -> func.HttpResponse:
    try: 
        logging.info('Python HTTP trigger function processed a request.')
        
        #Declaring headers for response
        headers = {}
        headers["Content-Type"] = "application/json"
        headers["Access-Control-Allow-Origin"] = "*"
        headers["Access-Control-Allow-Headers"] = "Authorization"
        req_body = req.get_json()
        if req_body is None:
            error = 'key is not found'
            return func.HttpResponse(f"{error}!", status_code=401, headers=headers)
        else:
            purpose = req_body['purpose']
        if purpose == 'prediciton':
            try:
                data = req_body['data']
                test_data = pd.DataFrame(data)
                output = prediction(test_data)
                dictt = {'output': output}
                return func.HttpResponse(dumps(dictt), status_code=200, headers=headers)
            except Exception as e:
                dictt = {"error": str(e)}
                return func.HttpResponse(dumps(dictt), status_code=401, headers=headers)
        elif purpose == 'training':
            key = req_body['key']
            status = train_model(key)
            dictt = {'status': status}
            return func.HttpResponse(dumps(dictt), status_code=200, headers=headers)
        elif purpose == 'evaluation':
            try:
                score = evaluate_model()
                dictt = {'score': score}
                return func.HttpResponse(dumps(dictt), status_code=200, headers=headers)
            except Exception as e:
                dictt = {"error": str(e)}
                return func.HttpResponse(dumps(dictt), status_code=401, headers=headers) 
    except Exception as e:
        status = {"message": "Error: " + str(e)}
        response_code = 501
        return func.HttpResponse(dumps(status), status_code=response_code, headers=headers)
