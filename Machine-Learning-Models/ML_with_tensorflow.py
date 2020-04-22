# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:03:35 2019

@author: Rajkumar
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, RFE
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.client import random_forest
from tensorflow.python.ops import resources
import gc
from scipy import ndimage
from subprocess import check_output
import cv2

def Feature_selection(df):
    #Backward Elimination
    X = df.iloc[:,1:12]
    y = df.iloc[:,12]
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

# load iris dataset
df1 = pd.read_csv('dataset/winequality-red.csv')
df1.columns = ['col']
df1 = pd.concat([df1,df1.col.str.split(';',expand=True)],1)
df2 = pd.read_csv('dataset/winequality-white.csv')
df2.columns = ['col']
df2 = pd.concat([df2,df2.col.str.split(';',expand=True)],1)
frames = [df1, df2]
df = pd.concat(frames)
df = df.reset_index(drop=True)

#iris = datasets.load_digits()
#x=iris.data
#y = iris.target
#x = pd.DataFrame(x)
#x.columns = iris.feature_names

# features & target
features = df.iloc[:,1:12]
features.columns = ['a','b','c','d','e','f','g','h','i','j','k']
features = features[['a','b','k','h','j']]
target = df.iloc[:,12:13]
target = pd.get_dummies(target).values
x = np.asanyarray(features)
y = np.asanyarray(target)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25,
                                                    random_state = 0)


num_features = X_train.shape[1]
num_labels = y_train.shape[1]

# Input and Target placeholders 
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int64, shape=[None, num_labels])

''' Neural network '''
num_hidden = int((num_features+num_labels)/2)
seed = 120

weights = {
    'hidden': tf.Variable(tf.random_normal([num_features, num_hidden], seed=seed)),
    'output': tf.Variable(tf.random_normal([num_hidden, num_labels], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([num_hidden], seed=seed)),
    'output': tf.Variable(tf.random_normal([num_labels], seed=seed))
}

hidden_layer = tf.add(tf.matmul(X, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.048, beta1=0.9,
                                   beta2=0.999, epsilon=1e-7).minimize(cost)
sess = tf.Session()
init = tf.global_variables_initializer()
epochs = 501

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)    
    for epoch in range(epochs):
        if epoch%100 == 0:
            _, c = sess.run([optimizer, cost], feed_dict = {X: X_train, Y: y_train})            
            avg_cost = c           
            print("Epoch: {0}, cost = {1}".format(epoch, avg_cost))    
            # find predictions on val set
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(pred_temp, tf.float32))*100
            print("Validation Accuracy: {} %".format(accuracy.eval({X: X_test, Y: y_test})))    
        optimizer.run(feed_dict={X: X_train, Y: y_train})
    print("\nTraining complete!") 
    predict = tf.nn.softmax(output_layer)
    pred = predict.eval({X: X_test})
    print("\nPrediction complete!")
sess.close()


#EPOCHS = 10
#model = build_model()
#
#model.fit(X_train, y_train, epochs=EPOCHS,
#          validation_split = 0.2, verbose=1)
#
#loss, mae, mse = model.evaluate(X_test, y_test, verbose=1)
#
#test_predictions = model.predict(X_test)

'''              '''


# Parameters
num_steps = 100 # Total steps to train
num_trees = 10 
max_nodes = 1000

# Random Forest Parameters
params = tensor_forest.ForestHParams(num_classes=num_labels,
                                      num_features=num_features, num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

#build the graph
graph_builder_class = tensor_forest.RandomForestGraphs
clf = random_forest.TensorForestEstimator(params, graph_builder_class=graph_builder_class)

# Get training graph and loss
train_op = graph_builder_class.training_graph(X, Y)
loss_op = graph_builder_class.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = graph_builder_class.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.int64))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training
for i in range(1, num_steps + 1):
    _, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))



# Test Model
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))

train_input_fn = tf.estimator.inputs.numpy_input_fn(x=X_train, y=y_train,
                                                    num_epochs=None, shuffle=True)
clf.fit(input_fn=train_input_fn, steps=500)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x=X_test,
                                                    num_epochs=None, shuffle=True)
result = clf.predict(tf.cast(X_test, tf.float32))
result = clf.predict(test_input_fn)




























