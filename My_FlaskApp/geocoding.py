# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:11:43 2020

@author: Rajkumar
"""
import os
import sys
import requests
import json
import pandas as pd
from threading import Thread
directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

import blob

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