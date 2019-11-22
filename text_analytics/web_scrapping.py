# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:46:06 2019

@author: Rajkumar
"""
import re
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen
import requests
import json

products=[] #List to store name of the product
prices=[] #List to store price of the product
ratings=[] #List to store rating of the product

html = urlopen("https://www.flipkart.com/laptops/~buyback-guarantee-on-laptops-/pr?sid=6bo%2Cb5g&uniq")
soup = BeautifulSoup(html, 'lxml')
print(soup.title)

for a in soup.findAll('a',href=True, attrs={'class':'_31qSD5'}):
    name=a.find('div', attrs={'class':'_3wU53n'})
    price=a.find('div', attrs={'class':'_1vC4OE _2rQ-NK'})
    rating=a.find('div', attrs={'class':'hGSR34'})
    products.append(name.text)
    prices.append(price.text)
    ratings.append(rating.text)
df = pd.DataFrame({'Product Name':products,'Price':prices,'Rating':ratings})

all_links = soup.find_all("a")
for link in all_links:
    print(link.get("href"))
    
div = soup.find_all('div', attrs={'class', 'k-tabstrip-wrapper'})
# extract the hotel name and price per room
for card in div:
    # get the hotel name
    hotel_name = card.find('p')
    # get the room price
    room_price = card.find('li', attrs={'class': 'htl-tile-discount-prc'})
    print(hotel_name.text, room_price.text)

def extract_table(url):
    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')
    rows = soup.find_all('tr')
    rows.pop(1)
    list_rows = []
    for row in rows:
        row_td = row.find_all('td')
        str_cells = str(row_td)
        clean = re.compile('<.*?>')
        clean2 = (re.sub(clean, '',str_cells))
        cleantext = BeautifulSoup(clean2, "lxml").get_text()
        list_rows.append(cleantext)
    df = pd.DataFrame(list_rows)
    df1 = df[0].str.split(',', expand=True)
#    df1 = df1.iloc[:,0:6]
    df1 = df1.drop(1,axis=1)
    df1 = df1.drop(0,axis=0)
    df1.columns = ['Month','Kolkata','Delhi',' Mumbai','Chennai']
    df1['Month'] = df1['Month'].str.replace('[','')
    df1['Chennai'] = df1['Chennai'].str.replace(']','')
#    col_labels = soup.find_all('th')
#    all_header = []
#    col_str = str(col_labels)
#    cleantext2 = BeautifulSoup(col_str, "lxml").get_text()
#    all_header.append(cleantext2)
#    df2 = pd.DataFrame(all_header)
#    header = df2[0].str.split(',', expand=True)
    return df1
url = "https://www.iocl.com/Product_PreviousPrice/PetrolPreviousPriceDynamic.aspx"
petrol = extract_table(url)
url = "https://www.iocl.com/Product_PreviousPrice/DieselPreviousPriceDynamic.aspx"
diesel = extract_table(url)
url = "http://www.fabpedigree.com/james/mathmen.htm"
html = urlopen(url)
url = "http://truckbhada.com/PostedLoadDetails?st=Maharashtra"
soup = BeautifulSoup(html, 'lxml')
dropdown=soup.select('li') 
for i, li in enumerate(dropdown):
        print(i, li.text)
name_box = soup.find('h1', attrs={'class': 'name'})
name_box.text.strip()
# find results within table
table = soup.find('table', attrs={'class': 'k-selectable'})
results = table.find_all('tr')
for result in results:
    rank = result.find_all('td')[0].getText()
# go to link and extract company website
    url = result.find_all('td').find('a').get('href')

# find all with the image tag
images = soup.find_all('img', src=True)
# select src tag
image_src = [x['src'] for x in images]
# select only jp format images
image_src = [x for x in image_src if x.endswith('.jpg')]
image_count = 1
for image in image_src:
    with open('image_'+str(image_count)+'.jpg', 'wb') as f:
        res = requests.get(image)
        f.write(res.content)
    image_count = image_count+1



# API url
url = "https://footballapi.pulselive.com/football/players"
url = "https://www.truckbhada.com/WS/rdXD4l0kD8.asmx/DailyLoadsNew"
# Headers required for making a GET request
# It is a good practice to provide headers with each request.

headers = {
#    "content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
"content-Type": "application/json",
    "DNT": "1",
    "Authorization":"Bearer{0}".format(token),
    "Origin": "https://transporter.lorri.in",
    "Referer": "https://transporter.lorri.in/generalInfo",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36"
}
token = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjhhMzY5M2YxMzczZjgwYTI1M2NmYmUyMTVkMDJlZTMwNjhmZWJjMzYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vdHBtcC10cmFuc3BvcnRlci1wcm9kdWN0aW9uIiwiYXVkIjoidHBtcC10cmFuc3BvcnRlci1wcm9kdWN0aW9uIiwiYXV0aF90aW1lIjoxNTczNjI4NDQ5LCJ1c2VyX2lkIjoieXlFanRuem12alk3UU1FQVk0NGlzWlNOcnh6MiIsInN1YiI6Inl5RWp0bnptdmpZN1FNRUFZNDRpc1pTTnJ4ejIiLCJpYXQiOjE1NzQ0MDgxMTgsImV4cCI6MTU3NDQxMTcxOCwicGhvbmVfbnVtYmVyIjoiKzkxMTIzMTIzNzg5NSIsImZpcmViYXNlIjp7ImlkZW50aXRpZXMiOnsicGhvbmUiOlsiKzkxMTIzMTIzNzg5NSJdfSwic2lnbl9pbl9wcm92aWRlciI6InBob25lIn19.qqXl1wPWTq2oW7sOaEFNi_9L-RcKXFcFCpZmVeQCVxIFdY8hTNYQqSCQ0KTyAZNDfsdR7rHugyENusZPQr7XhgNoLDqsXVbwV1r_wWyhbeIOTJuOpdFRbfD0fE1NdhzJDljdMoqgtlA_EFS4PQx5rZmMRYQvBg5xcfLTPzkgzVXBuVJ7H6nngdI9Qy4t7EmULtJgf4po9lzQ4g_Lj5T-vaITVmJ8Ad6prjLfDpnkTZDZA2dicjO4fpcmwjMee3L9BOhyisW_8OJ4GGdR2fLGCk_4nod0C3pa4U4uaXAJZcfiu_VHhKD4BDhXLn4kFYLp1NbJ-4_WofndpwWOOguNuQ"
# Query parameters required to make get request
queryParams = {"searchFields": "jkcement"}
    "pageSize": 32,
    "compSeasons": 274,
    "altIds": True,
    "page": 0,
    "type": "player",
    "id": -1,
    "compSeasonId": 274
}

# Sending the request with url, headers, and query params
response = requests.get(url = url, headers = headers,data = json.dumps(data))
                        , params = queryParams)
data = json.loads(response.text)
# if response status code is 200 OK, then
if response.status_code == 200:
    # load the json data
    data = json.loads(response.text)
    # print the required data
    for player in data["content"]:
        print({
            "name": player["name"]["display"],
            "nationalTeam": player["nationalTeam"]["country"],
            "position": player["info"]["positionInfo"]
        })

def get_data(list1):
    df = pd.DataFrame()
    for ind in list1:
        queryParams = {"state": ind}
        response = requests.get(url = url, headers = headers, params = queryParams)
        data = json.loads(response.text)
        data = pd.DataFrame(data)
        df = df.append(data)
        df = df.reset_index(drop=True)
    return df

url='https://www.truckbhada.com/WS/rdXD4l0kD8.asmx/RecentLoad1200NewThirtyDay'
url = 'https://www.truckbhada.com/WS/rdXD4l0kD8.asmx/RecentLoad1200NewSixtyDay'
url='https://apilorri.azurewebsites.net/api/my-profile'
list1=list(pd.read_excel('states.xlsx')['state'])
df = get_data(list1)
df.to_excel('21_11_19_dump.xlsx')





