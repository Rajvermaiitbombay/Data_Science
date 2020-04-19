# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:08:54 2020

@author: Rajkumar
"""
import pandas as pd
import json
from bson.json_util import dumps
from pymongo import MongoClient
'''
1. MongoDB stores data in JSON-like documents, which makes the database very flexible and scalable.
2. each collection has its unique structure. 
(client->database->collection->document)
'''

class doc:
    host = 'lncosmosdbtest.documents.azure.com'
    user = 'lncosmosdbtest'
    port = 10255
    passwd = ('tisybVwtEsHYlGycQa7yxHoUSs3aL2tLIqbdmQN2KYJ9oxobXtLI6c'
              + 'SHPNS4JNBlukbtFHGH7aEQrXplu9KtMw==')

class connection:
    def __init__(self, host, user, port, passwd):
        self.host = host
        self.user = user
        self.port = port
        self.passwd = passwd
    def connect_to_db(self, databasename):
        self.dbname = databasename
        uri = ('mongodb://%s:%s@%s:%d/?ssl=true&replicaSet=globaldb' %
               (self.user, self.passwd, self.host, self.port))
        self.client = MongoClient(uri)
        db = self.client[self.dbname]
        return db
        
    def documentCons(self, df):
        data = df.to_json(orient='records')
        data = json.loads(data)
        return data

mongo = connection(doc.host, doc.user, doc.port, doc.passwd)
mydb = mongo.connect_to_db('LogisticsNow')
collection = mydb['Log1']

'''Check the list of Databases in MongoDB '''
print(mongo.client.list_database_names())
'''Check the list of collections in Database '''
print(mydb.list_collection_names())

'''Insert data into collection '''
x = collection.insert_one(mydict)
x = collection.insert_many(mylist)
print(x.inserted_ids)

'''find/read data from collection '''
x = collection.find_one()
x = collection.find({},{ "_id": 0, "name": 1, "address": 1 })
x = collection.find({},{ "address": 0 })
x = list(collection.find({"activity" : "Transporter Data Update"}))
# s or greater than s (alphabetically)
myquery = { "address": { "$gt": "S" } }
myquery = {"age": {"$gte": 20}}
myquery = {'email':{"$exists":True} }
myquery = {"skills":{"$all":["mongodb","python"]}}
myquery = {"seconds":{"$ne":60}}
myquery = {"skills":{"$nin":["php","ruby","perl"]}}
myquery = {"age": {"$lt": 20}}
myquery = {"age": {"$lte": 20}}
myquery = {"employees": {"$in": ["emp1", "emp2"]} }
myquery = {"$or": [ {"age": 10}, {"intrests.name": "painting"} ]}
myquery = {"$nor": [ {"age": 10}, {"intrests.name": "painting"} ]}
query = {'$and': [{'field 1': 'MUST MATCH THIS'}, {'field 2': 'painting'}]}
# Array Must Be Of Size
query = {"skills":{"$size":3} }


# not equal to
myquery = {"age": {"$ne": 20}}
myquery = { "address": { "$regex": "^S" } }
mydoc = collection.find().sort("name", -1)
myresult = collection.find().limit(5)

'''Delete data from collection'''
collection.delete_one({ "address": "Mountain 21" })
x = collection.delete_many({ "address": {"$regex": "^S"} })
ret = collection.delete_many({"category": "general"})
x = collection.delete_many({})

'''Update Data in collection '''
myquery = { "address": "Valley 345" }
newvalues = { "$set": { "address": "Canyon 123" } }
collection.update_one(myquery, newvalues)
myquery = { "address": { "$regex": "^S" } }
newvalues = { "$set": { "name": "Minnie" } }
x = collection.update_many(myquery, newvalues)

''' drop the collection '''
collection.drop()



































