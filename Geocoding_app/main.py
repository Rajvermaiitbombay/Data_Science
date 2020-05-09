# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:22:53 2020

@author: Rajkumar
"""

#importing libraries
import os
import sys
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file, Response
import requests
from threading import Thread
import io

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

import blob
directory = 'tmp/'
blob_url = 'https://logisticsnowtech3.blob.core.windows.net/rajkumar/'

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

'''Generate geocoding from address '''
def generate(address):
    endpoint = 'https://maps.googleapis.com/maps/api/geocode/json'
    data = {"address": address, "key": "AIzaSyB76IJyqObJgFJvVPtmENhyby-k8R0lN_s"}
    response = requests.get(endpoint, params = data)
    output = json.loads(response.content)
    result = output['results'][0]
    right_address  = result['formatted_address']
    geometry  = result['geometry']['location']
    lat = geometry['lat']
    lon = geometry['lng']
    dfh = pd.DataFrame({'location': [address], 'lat': [lat], 'lon': [lon], 'address': [right_address]})
    return dfh

def geocoding(list1):
    df = pd.DataFrame()
    for ind in list1:
        try:
            dfh = generate(ind)
            df = df.append(dfh)
        except Exception as e:
            print(str(e))    
    return df

def mail_fun(data):
    list1 = list(data['Full Address'])
    threads = [None] * 3
    threads[0] = ThreadWithReturnValue(target=geocoding, args=(list1[:200],))
    threads[1] = ThreadWithReturnValue(target=geocoding, args=(list1[200:400],))
    threads[2] = ThreadWithReturnValue(target=geocoding, args=(list1[400:],))
    threads[0].start()
    threads[1].start()
    threads[2].start()
    df1 = threads[0].join()
    df2 = threads[1].join()
    df3 = threads[2].join()
    df = pd.concat([df1, df2, df3], axis=0)
    df = df.reset_index(drop=True)
    output = data.merge(df, left_on='Full Address', right_on="location", how='inner')
    output = output.drop('location', axis=1)
    try:
        output.columns = ['Ward #', 'Sr. No', 'Pin Code', 'Pincode Address', ' Area Type',
                          'Full Address', 'latitue', 'longlitude', 'Right Address']
    except Exception:
        pass
    blob.writeBlob_text(output, 'output.csv', container_name='rajkumar')
    return None
        
    
UPLOAD_FOLDER = 'tmp/'
#creating instance of the class
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#to tell flask what url shoud trigger the function index()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def prog():
    return render_template('signupform.html')
@app.route('/logout')
def logout():
    return render_template('signupform.html')

@app.route('/signup', methods = ['GET','POST'])
def signup():
    if request.method == 'GET':
        return render_template('notify.html')
    else:
        name = request.form['username']
        email = request.form['email']
        password = request.form['pass']
        signup = pd.read_csv(blob_url+'signup.csv')
        signup = signup.drop('idx', axis=1)
        if signup[signup["username"]==name].empty == False:
            error = 'Username already exist' 
            return render_template('signupform.html', error=error)             
        add = pd.DataFrame({'username':[name],'email_id':[email],'password':[password]})
        signup = signup.append(add)
        blob.writeBlob_text(signup, 'signup.csv', container_name='rajkumar')
        return render_template('signupform.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    info = pd.read_csv(blob_url+'signup.csv')
    user = request.form['username']
    password = request.form['password']
    if info[info["username"]==user].empty == True and info[info["password"]==password].empty == True:
        error = 'username and password, both are incorrect!' 
        return render_template('signupform.html',error=error) 
    elif info[info["username"]==user].empty == True:
        error = 'username incorrect!' 
        return render_template('signupform.html',error=error)
    elif password != info[info["username"]==user].set_index('username')["password"][user] and info[info["username"]==user].empty == False:
        error = 'password incorrect!' 
        return render_template('signupform.html',error=error)
    elif password == info[info["username"]==user].set_index('username')["password"][user]: 
        return render_template('upload.html') 
    else:
        return render_template('signupform.html')

@app.route('/upload')
def upload():
    if request.method == 'GET':
        return render_template('notify.html')
    return render_template('upload.html')

# result page
@app.route('/result', methods=['GET', 'POST'])
def result():
    global file
    if request.method == 'GET':
        return 'Sorry ! we did not get any dataset'
    elif request.method == 'POST' and request.files['myfile']:
        f = request.files['myfile']
        file = os.path.splitext(f.filename)[-1].lower()
        if file == ".xlsx":
            test_data = pd.read_excel(f, encoding='utf-8')
        elif file == ".csv":
            test_data = pd.read_csv(f, encoding='latin1')
        mail_fun(test_data)
    return render_template('result.html')

@app.route('/download', methods=['POST'])
def format_download():
    data = pd.read_csv(blob_url+'output.csv')
    file_stream = io.StringIO()
    file_stream=data.to_csv(index_label="idx",encoding="utf-8")
    return Response(file_stream,
                    mimetype="text/csv",
                    headers={"Content-disposition":
                        "attachment; filename=output.csv"})

if __name__ == '__main__':
    app.run(port=8766)
