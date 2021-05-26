from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus

import pandas as pd
import xmltodict
import json


key='__key__'
url = f'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson?serviceKey={key}&'
queryParams = urlencode({ quote_plus('pageNo') : 1, 
                          quote_plus('numOfRows') : 10,
                          quote_plus('startCreateDt') : '20210119',
                          quote_plus('endCreateDt') : '202105250'})
url2 = url + queryParams
response = urlopen(url2)
# print(type(response)) # HTTPSresponse 
results = response.read().decode("utf-8")
# print(type(results))   # str
results_to_json = xmltodict.parse(results)
data = json.loads(json.dumps(results_to_json))
print(type(data))   # dic
print(data)

##----------------------------

datas = data['response']['body']['items'].values()

data_list = []

def c_info():
    for data in datas:
        for i in data:
            info = {           
                'createDt' : ''.join(i['createDt'].split()[0].split('-')),
                'decideCnt' : i['decideCnt'],
                'clearCnt' : i['clearCnt'],
                'careCnt' : i['careCnt'],
                'deathCnt' : i['deathCnt']           
            }       
            data_list.append(info)
    return data_list

c_info()  

df = pd.DataFrame(data_list)
df.columns = ['날짜','누적확진자','격리해제환자','치료중환자','사망자수']
df = df.sort_values(by='날짜', ascending=True)
df.set_index('날짜', inplace=True)

# df.to_csv("./df_covid19.csv")



##----------------------------

covid = []
for data in datas:
    for i in data:
        createDt = ''.join(i['createDt'].split()[0].split('-'))
        decideCnt = i['decideCnt']
        clearCnt = i['clearCnt']
        careCnt = i['careCnt']
        deathCnt = i['deathCnt']  
        covid_a = createDt, decideCnt, clearCnt, careCnt, deathCnt
        covid_a = list(covid_a)
        covid.append(covid_a)

df = pd.DataFrame(covid)
df.columns = ['날짜','누적확진자','격리해제환자','치료중환자','사망자수']
df = df.sort_values(by='날짜', ascending=True)
df.set_index('날짜', inplace=True)


##----------------------------Review

corona=data['response']['body']['items']['item']

#추가하고 싶은 리스트 생성
Date=[]
Cnt=[]
clear_cnt=[]
care_cnt=[]
death_cnt=[]
exam_cnt=[]     # examCnt   검사중

for i in corona:
    Date.append(i['stateDt'])  #'stateDt': '20200801'
    Cnt.append(i['decideCnt'])  # decideCnt': '14336'   누적확진자
    clear_cnt.append(i['clearCnt'])   # 13233           격리 해제환자
    care_cnt.append(i['careCnt'])     # 802             치료중 환자
    death_cnt.append(i['deathCnt'])    #301             사망자 수

df=pd.DataFrame([Date,Cnt,clear_cnt,care_cnt,death_cnt]).T
df.columns=['날짜','누적확진자','격리해제환자','치료중환자','사망자수'] 
df=df.sort_values(by='날짜', ascending=True)
df.set_index('날짜', inplace=True)

df.to_csv('sample.csv')