# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:22:53 2020

@author: Rajkumar
"""

#importing libraries
import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, Response
import io
import json

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

import blob
import geocoding

path = os.getcwd()
directory = 'tmp/'
blob_url = 'https://logisticsnowtech3.blob.core.windows.net/rajkumar/'

#creating instance of the class
app=Flask(__name__)

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
        geocoding.mail_fun(test_data)
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
    app.run(port=6969)
