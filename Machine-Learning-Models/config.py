# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:05:40 2019

@author: Rajkumar
"""
import sys
import os

directory = os.path.dirname(__file__)
sys.path.insert(0,directory)

class sql:
    user = "logisticsnow@datacleaningdb"
    password = "Mudjm5vj"
    host = "datacleaningdb.mysql.database.azure.com"
    port = 3306


# ------


class tinadocumentdb:
    host = 'lncosmosdbtest.documents.azure.com'
    user = 'lncosmosdbtest'
    port = 10255
    passwd = ('tisybVwtEsHYlGycQa7yxHoUSs3aL2tLIqbdmQN2KYJ9oxobXtLI6cSHPN'
              + 'S4JNBlukbtFHGH7aEQrXplu9KtMw==')


class dcdocumentdb:
    host = 'datacleaningdatabase.documents.azure.com'
    user = 'datacleaningdatabase'
    port = 10255
    passwd = ('mBABq1LKSrSqi9jHhFoTtisafiagcw63ZhJPIj5RF6RTNEs4Ptan1'
              + 'Tsnnb6hEq2FqMfSZiA8HQzCi2lhRnAtlw==')

class blobcred:
    account_name = 'logisticsnowtech3'
    account_key = ('FIo9u4zXxDzT4ZEwwcZMBnrKvhQthUBwEqsD9NFI0owgCVdf6FNpO9zKJkE9yh72PVXHjclA28C0H0xRE1j1TA==')