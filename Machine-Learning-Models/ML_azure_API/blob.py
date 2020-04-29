"""
 Created on Fri Apr 19 20:47:07 2019

 @author: Rajkumar
 """

from azure.storage.blob import BlockBlobService
import pandas as pd
import io
import os
import re
import sys
from sklearn.externals import joblib

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

from config import blobcred as c

def connInitate():
    block_blob_service = BlockBlobService(c.account_name, c.account_key)
    return block_blob_service


def writeBlob(file_path, filename, container_name='dcrawfile'):
    block_blob_service = connInitate()
    block_blob_service.create_blob_from_path(container_name, r'{}'.format(filename),
                                             r'{}'.format(file_path))
    return print("Thanks")


def filePrint(container_name='dcrawfile'):
    block_blob_service = connInitate()
    print("\nList blobs in the container")
    generator = block_blob_service.list_blobs(container_name)
    for blob in generator:
        print("\t Blob name: " + blob.name)


def getBlob(blob_name_incont, download_path, container_name='dcrawfile'):
    block_blob_service = connInitate()
    block_blob_service.get_blob_to_path(container_name, blob_name_incont,
                                        download_path)

def _get_random_bytes(size):
    import random
    rand = random.Random()
    result = bytearray(size)
    for i in range(size):
        result[i] = rand.randint(0, 255)
    return bytes(result)
    
def getBlob_url(dataset_id, url, container_name='dcrawfile'):
    block_blob_service = connInitate()
    my_stream_obj = io.BytesIO()
    extension = os.path.splitext(url)[-1].lower()
#    blob_name_incont = "%s_%sraw.xlsx" % (dataset_id, Filename)
    blob_name_incont = re.split('/',url)[-1]
    block_blob_service.get_blob_to_stream(container_name, blob_name_incont,
                                          my_stream_obj)
    my_stream_obj.seek(0)
#    df=pd.read_excel(my_stream_obj, encoding='latin1')
#    path = "https://logisticsnowtech3.blob.core.windows.net/dcrawfile/"
#    Filename = os.path.splitext(file)[0].lower()
#    raw_filename = "%s_%sraw.xlsx" % (dataset_id, Filename)
#    blob_path = path + raw_filename
    if extension == ".xlsx":
        df = pd.read_excel(my_stream_obj, encoding='latin1')
    elif extension == ".csv":
        df = pd.read_csv(my_stream_obj, encoding='latin1')
    elif extension == ".xls":
        xls = pd.ExcelFile(my_stream_obj)
        df = xls.parse(0)
    return df

def getBlob_stream(blob_name_incont, container_name='dcrawfile'):
    block_blob_service = connInitate()
    my_stream_obj = io.BytesIO()
#    blob_name_incont = "%s_%sraw.xlsx" % (dataset_id, filename)
    block_blob_service.get_blob_to_stream(container_name, blob_name_incont,
                                          my_stream_obj)
    my_stream_obj.seek(0)
    model = joblib.load(my_stream_obj)
    return model

def writeBlob_stream(blob_name_incont, filename, container_name='dcrawfile'):
    file_stream = io.BytesIO(blob_name_incont)
    block_blob_service = connInitate()
    block_blob_service.create_blob_from_stream(container_name, filename,
                                                      file_stream)
    print('pass')
    return None
