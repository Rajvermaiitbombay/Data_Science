# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:16:56 2018

@author: Rajkumar
"""
import os
os.chdir(os.path.dirname(__file__))
import threading
import webbrowser 
import random
import pandas as pd
import numpy as np
import flask
import pickle,json
from flask import Flask, render_template, url_for,request, jsonify,flash,send_file,Response, session, abort,json
import sklearn
from sklearn.externals import joblib
import glob
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from werkzeug.serving import run_simple
from sqlalchemy import create_engine
import mysql.connector as sql
# Read CSV file #####
lsp = pd.read_csv('lsp.csv')
cord = pd.read_excel('cord_maha.xlsx')
## Split the training and tesing datasets from main datasets
train = lsp.iloc[np.r_[1:11,16:21],3:10]

## import support vector machine #####
from sklearn.svm import SVC
svc= SVC(kernel='rbf')
svc.fit(train.iloc[:,0:6],train.iloc[:,6])
pickle.dump(svc,open("lsp_rfc.pkl","wb"))
my_model = pickle.load(open("lsp_rfc.pkl","rb"))
    
app=Flask(__name__,template_folder='templates')

@app.route('/')
def student():
   return render_template('index.html')
@app.route('/upload')
def upload():
	return render_template('upload.html')
@app.route('/log')
def log():
    return render_template('login.html')
@app.route('/main')
def main():
    return render_template('mainpage.html')
@app.route('/data')
def data():
    return render_template('data.html')
@app.route('/home')
def toggle():
    return render_template('table.html')
@app.route('/register')
def prog():
    return render_template('signupform.html')
@app.route('/logout')
def logout():
    return render_template('signupform.html')
@app.route('/chat')
def chat():
    return render_template('chatbots.html')
@app.route('/bar')
def bar():
    lsp = pd.read_excel("lsp.xlsx")
    lname=list(lsp.Name)
    lsp1=lsp.iloc[:,1:9]
    cord1=cord.to_dict('dict')
    json1 = json.dumps(cord1)
    di=lsp1.set_index('Name').T.to_dict('list')
    jsondf = json.dumps(di)
    json2 = json.dumps(cord1)
    lsp2=lsp1.reset_index(drop=True)
    lsp2.set_index('Name', inplace=True)
    age=list((lsp[lsp['Name']=='a'].set_index('Name').transpose()).iloc[1:8,:]['a'])
    name=list(lsp2.columns)
    return render_template('bar.html',age=age,name=name,lname=lname,df=jsondf,cord=json1,cord1=json2)

test=pd.DataFrame()
@app.route('/signup', methods = ['GET','POST'])
def signup():
    if request.method == 'GET':
        return render_template('notify.html')
    else:
        name = request.form['username']
        email = request.form['email']
        password = request.form['pass']
        engine = create_engine('mysql+mysqlconnector://root:root@localhost/dummy',echo = False)
        signup = pd.DataFrame({'username':[name],'email_id':[email],'password':[password]})
        signup.to_sql(name='signup',con=engine,if_exists='append',index=False)
        return render_template('signupform.html')
    
@app.route('/login', methods = ['GET','POST'])
def login():
    user = request.form['username']
    password = request.form['password']
    conn = sql.connect(host ='localhost' ,database ='dummy' ,user = 'root',password = 'root')
    cursor = conn.cursor()
    query1 = ('SELECT * FROM signup')
    cursor.execute(query1) 
    info = cursor.fetchall()
    info = pd.DataFrame(info)
    info.columns = ['username','email_id','password']
    conn.close()
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
        return render_template('table.html') 
    else:
        return render_template('signupform.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    global test
    if request.method == 'GET':
        return render_template('notify.html')
    elif request.method == 'POST' and request.files['myfile']:
        f = request.files['myfile']
        test = pd.read_csv(f)
        df = my_model.predict(test.iloc[:,2:8])
        dt =pd.DataFrame(df)
        test1=test.reset_index(drop=True)
		## concatnate two dataframes into one dataframe
        total=pd.concat([test1,dt],axis=1)
        total.columns = ['Name','Date','x','y','z','xi','yi','zi','prediction']
        return render_template('data.html')
@app.route('/download') # this is a job for GET, not POST
def data_csv():
    return send_file('pred.xlsx',
                     mimetype='xlsx/csv',
                     attachment_filename='pred.xlsx',
                     as_attachment=True)
@app.route('/result')
def result():
    return render_template('table.html',  tables=[test.to_html(index=False)], titles=test.columns.values)
@app.route('/test')
def test():
    Name=test['Name'].values.tolist()
    return render_template('test.html', tables=test.values.tolist(), name=Name)

if __name__ == '__main__':
    port=5000+ random.randint(0,999)
    url="http://127.0.0.1:{0}".format(port)
    threading.Timer(1.25,lambda:webbrowser.open(url)).start()
    run_simple('127.0.0.1',port, app,use_debugger=True, use_evalex=True)
