# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:44:17 2020

@author: Rajkumar
"""
import json
import boto3
from boto3.dynamodb.conditions import Key, Attr
#from jwt_rsa.token import JWT
#from jwt_rsa.rsa import generate_rsa

#def jwt_decoder(token):
#    try:
#        bits = 2048
#        private_key, public_key = generate_rsa(bits)
#        jwt = JWT(private_key, public_key)
#        result = jwt.decode(token, verify=False)
#        return result
#    except Exception as e: 
#        return e

def lambda_handler(event, context):
    table_name = event['table_name']
    action = event['action']
    if action == 'fetch':
        key = event['key']
        value = event['value']
        db = Database_Operations()
        table = db.dynamodb_initialization(table_name)
        records = db.dynamodb_getitem(table, key, value)
        return {'statusCode': 200, 'body': json.dumps(records)}
    elif action == 'dump':
        data = event['data']
        db = Database_Operations()
        table = db.dynamodb_initialization(table_name)
        status = db.dynamodb_putitem(table, data)
        return {'statusCode': 200, 'body': json.dumps(status)}
    elif action == 'drop':
        key = event['key']
        value = event['value']
        db = Database_Operations()
        table = db.dynamodb_initialization(table_name)
        status = db.dynamodb_deleteitem(table, key, value)
        return {'statusCode': 200, 'body': json.dumps(status)}
    elif action == 'fetchAll':
        db = Database_Operations()
        table = db.dynamodb_initialization(table_name)
        records = db.dynamodb_getallitem(table)
        return {'statusCode': 200, 'body': json.dumps(records)}
    elif action == 'scanString':
        key = event['key']
        value = event['value']
        db = Database_Operations()
        table = db.dynamodb_initialization(table_name)
        records = db.dynamodb_scanstring(table, key, value)
        return {'statusCode': 200, 'body': json.dumps(records)}

class Database_Operations:
    def dynamodb_initialization(self, table_name):
        try:
            dynamodb = boto3.resource('dynamodb')
            table = dynamodb.Table(table_name)
            return table
        except Exception as e: 
            return str(e)
    
    def dynamodb_putitem(self, table, data):
        try:
            table = table
            for item in data:
                response = table.get_item(Key={ 'email': item['email']})
                if 'Item' not in response:
                    table.put_item(Item = item)
            return 'Item insert successful'
        except Exception as e: 
            return str(e)
    
    def dynamodb_getitem(self, table, key, value):
        try:
            response = table.get_item(Key={ key: value})
            if 'Item' in response:
                item = response['Item']
                return item
            else:
                return "No Records found"
        except Exception as e: 
            return str(e)
    
    def dynamodb_deleteitem(self, table, key, value):
        try:
            table.delete_item(Key={ key: value})
            return 'Item deletion successful'
        except Exception as e: 
            return str(e)

    def dynamodb_getallitem(self, table):
        try:
            response = table.scan()
            items = response['Items']
            return items
        except Exception as e: 
            return str(e)
        
    def dynamodb_scanstring(self, table, key, value):
        try:
            response = table.scan(FilterExpression=Attr(key).contains(value))
            '''begins_with, eq, contains '''
            items = response['Items']
            return items
        except Exception as e: 
            return str(e)
    
    def dynamodb_scannumber(self, table, key, value):
        try:
            response = table.scan(
                FilterExpression=Attr(key).lt(value)
            )
            items = response['Items']
            return items
        except Exception as e: 
            return str(e)