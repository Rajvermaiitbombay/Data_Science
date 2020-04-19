# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:20:50 2019

@author: Rajkumar
"""
import pandas as pd
from sqlalchemy import create_engine
import variable as v
from config import sql as c
import json

database = "data_cleaning"
#database = "trucktypedc"
connection = "mysql+mysqlconnector://{0}:{1}@{2}:{3}/{4}".format(c.user,
                                                                 c.password,
                                                                 c.host, c.port,
                                                                 database)
tableName = "loc_master"
client_id = "100"


def listToString(df):
    if {v.tool_sugg}.issubset(df.columns):
        for i in range(0, len(df)):
            val = df[v.tool_sugg][i]
            type(val)
            print(val)
            df[i, v.tool_sugg] = json.dumps(val)
    if {v.pinlist}.issubset(df.columns):
        for i in range(0, len(df)):
            val = df[v.pinlist][i]
            type(val)
            print(val)
            df[i, v.pinlist] = json.dumps(val)
    return df


def stringConversion(df):
    if {v.tool_sugg}.issubset(df.columns):
        for i in range(len(df)):
            df[v.tool_sugg] = df[v.tool_sugg].astype(str)
    if {v.pinlist}.issubset(df.columns):
        for i, val in enumerate(df[v.pinlist]):
            df[v.pinlist] = df[v.pinlist][i].astype(str)
    return df



def stringToList(df):
    if {v.tool_sugg}.issubset(df.columns):
        for i in range(len(df)):
            val = df[v.tool_sugg][i]
            df[i, v.tool_sugg] = json.loads(val)
    if {v.pinlist}.issubset(df.columns):
        for i, val in enumerate(df[v.pinlist]):
            df[i, v.pinlist] = json.loads(df[v.pinlist][i])
    return df


def engine(connection):
    engine = create_engine(connection)
    return engine


def dumpTable(conn, df, tableName, client_id):
    trans = conn.begin()
    try:
        query = "DELETE FROM {0} WHERE dataset='{1}'".format(tableName,
                                                             client_id)
        conn.execute(query)
        print('client_id is not present')
        df = stringConversion(df)
        if {v.network_id}.issubset(df):
            pass
        else:
            df[v.network_id] = client_id
        df.to_sql(tableName, con=conn, if_exists='append', index=False)
        trans.commit()
    except NameError:
        trans.rollback()
        raise
    return None


def dumpAltTable(conn, df, tableName):
    df.to_sql(tableName, con=conn, if_exists='append', index=False)
    return None


def dumpManual(conn, df, tableName, client_id):
    trans = conn.begin()
    try:
        for i in range(len(df)):
            query = ('UPDATE {0} SET Map="{2}", manual_check=1 WHERE'
                     + ' (dataset="{1}" && `index`={3});').format(tableName, client_id,
                                                        df.Map[i], df['index'][i])
            conn.execute(query)
        trans.commit()
        return 'done'
    except Exception as e:
        trans.rollback()
        return str(e)


def dumpKeyword(conn, keyword, tableName):
    key = pd.DataFrame({'keyword': keyword})
    key.to_sql(tableName, con=conn, if_exists='append', index=False)
    return None


def dumpFilename(conn, filename, tableName):
    key = pd.DataFrame({'filename': [filename]})
    key.to_sql(tableName, con=conn, if_exists='append', index=False)
    return None


def fetchTable(conn, client_id, tableName):
    query = "SELECT * from {0} WHERE dataset='{1}';".format(tableName,
                                                            client_id)
#    query_col = 'select column_name from information_schema.columns where table_name = "{}"'.format(tableName)
    df = conn.execute(query).fetchall()
    if df == []:
        pass
    else:
        Columns = df[0].keys()
        df = pd.DataFrame(df)
        df.columns = Columns
    return df


def fetch_column(conn, column_name, tableName):
    try:
        query = "SELECT DISTINCT {0} FROM {1}".format(column_name, tableName)
        array = conn.execute(query).fetchall()
        array = list(array)
        col_name = [i[0] for i in array]
        return col_name
    except Exception as e:
        print(e)
        return []


def fetchMaster(conn, client_id, tableName):
    query = "SELECT * from {0} WHERE (dataset='{1}' && Map is null) or (Map='' && dataset='{1}');".format(tableName,
                                                                                                    client_id)
    query_col = 'select column_name from information_schema.columns where table_name = "{}"'.format(tableName)
    df = conn.execute(query).fetchall()
    if df == []:
        pass
    else:
        df = pd.DataFrame(df)
        col = conn.execute(query_col).fetchall()
        Columns = [i[0] for i in col]
        df.columns = Columns
    return df



def fetchMetadata(conn, tableName, userid):
    query = "SELECT * from {0} WHERE userid='{1}';".format(tableName, userid)
#    query_col = 'select column_name from information_schema.columns where table_name = "{}"'.format(tableName)
    result_proxy = conn.execute(query).fetchall()
    if result_proxy == []:
        return result_proxy
    else:
        df = pd.DataFrame(result_proxy)
        columns = result_proxy[0].keys()
        df.columns = columns
        return df

def fetch_filename(conn, column_name, tableName, userid):
    try:
        query = "SELECT DISTINCT {0} FROM {1} where userid='{2}';".format(column_name,
                                                                          tableName, userid)
        array = conn.execute(query).fetchall()
        array = list(array)
        col_name = [i[0] for i in array]
        return col_name
    except Exception as e:
        print(e)
        return []
def fetchUser(conn, tableName):
    query = "SELECT * from {0};".format(tableName)
#    query_col = 'select column_name from information_schema.columns where table_name = "{}"'.format(tableName)
    result_proxy = conn.execute(query).fetchall()
    if result_proxy == []:
        return result_proxy
    else:
        df = pd.DataFrame(result_proxy)
        columns = result_proxy[0].keys()
        df.columns = columns
        return df

def updateEntry(conn, tablename, colName, value, datasetid):
    query = 'UPDATE {0} SET {1} = "{2}" WHERE dataset="{3}";'.format(tablename,
                                                                 colName, value,
                                                                 datasetid)
    trans = conn.begin()
    try:
        conn.execute(query)
        trans.commit()
    except NameError:
        trans.rollback()
        raise
    return None
