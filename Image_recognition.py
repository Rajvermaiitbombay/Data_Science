# -*- coding: utf-8 -*-
"""
Created on Sun May  19 15:16:56 2019

@author: Rajkumar
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
os.chdir('C:/Rajkumar/new_folder')

# function that helps to fetch all images name from given folder
def openfile(folder):
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print("Working with {0} images".format(len(onlyfiles)))
    return onlyfiles

# Open all images
defect = openfile("YE358311_defects")
healthy = openfile("YE358311_Healthy")

# append all images into arrays (input data & labels)
from keras.preprocessing.image import img_to_array, load_img
healthy_data = []
for i in healthy:
    healthy_data.append(i)
healthy_label = ['healthy']*len(healthy_data)
defect_data = []
for i in defect:
    defect_data.append(i)
defect_label = ['defect']*len(defect_data)
data = healthy + defect
data_label = healthy_label + defect_label

# Exploratory data analysis
image_width = 640
image_height = 480
img = load_img("YE358311_Healthy" + "/" + healthy[2])
img.thumbnail((image_width, image_height), Image.ANTIALIAS)
x = img_to_array(img)
plt.imshow(x)

img = load_img("YE358311_defects" + "/" + defect[2])
img.thumbnail((image_width, image_height), Image.ANTIALIAS)
x = img_to_array(img)
plt.imshow(x)

# Data pre-processing
ratio = 4
image_width = int(image_width / ratio)
image_height = int(image_height / ratio)
channels = 3
nb_classes = 1
dataset = np.ndarray(shape=(len(data), image_height, image_width, channels),
                     dtype=np.float32)

# convert images into arrays and dump arrays into dataset
i = 0
for file in healthy:
    img = load_img("YE358311_Healthy" + "/" + file)  # this is a PIL image
    img.thumbnail((image_width, image_height), Image.ANTIALIAS)
    x = img_to_array(img)
    if x.shape == (120, 90, 3):
        img = img.resize((image_width, image_height), Image.ANTIALIAS)
        x = img_to_array(img)
    # Normalize
    x = (x - x.mean()) / x.std()
    dataset[i] = x
    i += 1
print("All healthy images converted to array & dumped into dataset!")
i = 139
for file in defect:
    img = load_img("YE358311_defects" + "/" + file)  # this is a PIL image
    img.thumbnail((image_width, image_height), Image.ANTIALIAS)
    x = img_to_array(img)
    if x.shape == (120, 90, 3):
        img = img.resize((image_width, image_height), Image.ANTIALIAS)
        x = img_to_array(img)
    # Normalize
    x = (x - x.mean()) / x.std()
    dataset[i] = x
    i += 1
print("All defect images converted to array & dumped into dataset!")

#Splitting dataset into training & testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, data_label, test_size=0.2, random_state=33)

# encoding categorical variables into binary variables
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

lb_make = LabelEncoder()
y_test = lb_make.fit_transform(y_test)
y_train = lb_make.fit_transform(y_train)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# find img rows, cols & colors
img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
colors = X_train.shape[3]

# Building the model
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

input_shape = (img_rows, img_cols, colors)
chanDim = -1
model = Sequential()
# (CONV => RELU => POOL) * 1
model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(AveragePooling2D(pool_size=(2, 2)))

# (CONV => RELU => POOL) * 2
model.add(SeparableConv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(SeparableConv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(AveragePooling2D(pool_size=(2, 2)))

# (CONV => RELU => POOL) * 3
model.add(SeparableConv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(SeparableConv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(SeparableConv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(AveragePooling2D(pool_size=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(2))
model.add(Activation("softmax"))
model.summary()

# compile model using accuracy to measure model performance
adam = keras.optimizers.Adam(lr=0.00146, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# gridsearch to select best epochs & batch size
def Gridsearch_epoch_batchSize(model):
    model = KerasClassifier(build_fn=model, verbose=0)
    batch_size = [10, 20, 40, 50, 60, 80, 100]
    epochs = [5, 10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return None


# gridsearch to select best optimizer
def Gridsearch_optimizer(model):
    model = KerasClassifier(build_fn=model, epochs=5, batch_size=50, verbose=0)
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return None


# gridsearch to select best learning rate
def Gridsearch_LearningRate(model):
    model = KerasClassifier(build_fn=model, epochs=5, batch_size=50, verbose=0)
    learning_rate = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 5]
    param_grid = dict(lr=learning_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return None


# gridsearch to select best Dropout Rate
def Gridsearch_Dropout_rate(model):
    model = KerasClassifier(build_fn=model, epochs=5, batch_size=50, verbose=0)
    dropout_rate = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return None


# train the model
model.fit(X_train, y_train, batch_size=50, epochs=5, verbose=1,
          validation_data=(X_test, y_test))

# check the accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#predict first 1 image in the test set
model.predict(X_test[:1])

# create a python web application to upload image and predict the result

UPLOAD_FOLDER = 'C:\\Rajkumar\\new_folder\\testing'
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# this function trigger the model
def prediction(file, model):
    img = load_img(file)
    img.thumbnail((image_width, image_height), Image.ANTIALIAS)
    x = img_to_array(img)
    if x.shape == (120, 90, 3):
        img = img.resize((image_width, image_height), Image.ANTIALIAS)
        x = img_to_array(img)
    # Normalize
    x = (x - x.mean()) / x.std()
    dataset[0] = x
    image = dataset[0].reshape(1, img_rows, img_cols, colors)
    result = model.predict(image)
    output = list(result[0])
    if (output[0] < output[1]):
        result1 = 'Healthy'
    elif (output[0] > output[1]):
        result1 = 'Cracked'
    else:
        result1 = 'Can not predict'
    return result1


# upload page
@app.route('/')
def upload():
    return render_template('upload.html')


# result page
@app.route('/result', methods=['GET', 'POST'])
def result():
    global file
    if request.method == 'GET':
        return 'Sorry ! we did not get any image'
    elif request.method == 'POST' and request.files['myfile']:
        file = request.files['myfile']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    file = 'C:/Rajkumar/new_folder/testing/' + filename
    result = prediction(file, model)
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run('127.0.0.1', 8765, use_evalex=True)
