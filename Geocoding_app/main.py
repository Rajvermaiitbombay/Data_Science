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
import io

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

import blob
directory = 'tmp/'
blob_url = 'https://logisticsnowtech3.blob.core.windows.net/rajkumar/'


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

def mail_fun(data):
    list1 = list(data['Full Address'])
    df = pd.DataFrame()
    for ind in list1:
        try:
            dfh = generate(ind)
            df = df.append(dfh)
        except Exception as e:
            print(str(e))    
    df = df.reset_index(drop=True)
    output = data.merge(df, left_on='Full Address', right_on="location", how='inner')
    output = output.drop('location', axis=1)
    output.columns = ['Ward #', 'Sr. No', 'Pin Code', 'Pincode Address', ' Area Type',
                      'Full Address', 'latitue', 'longlitude', 'Right Address']
    blob.writeBlob_text(output, 'output.csv', container_name='rajkumar')
    return None
        
    
UPLOAD_FOLDER = 'tmp/'
#creating instance of the class
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#to tell flask what url shoud trigger the function index()
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def form():
    if request.method == 'GET':
        return render_template('notify.html')
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            return render_template('upload.html')
        elif username == 'admin' and password != 'admin':
            error = 'Invalid Credentials. Please try again.'
            return render_template('login.html', error=error)
        elif username != 'admin' and password == 'admin':
            error = 'Invalid Credentials. Please try again.'
            return render_template('login.html', error=error)
        elif username != 'admin' and password != 'admin':
            error = 'Invalid Credentials. Please try again.'
            return render_template('login.html', error=error)


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
