
import numpy as np
import pandas as pd
path = "C:\\Users\\yjr\\Desktop\\temproot.csv"
data_set = pd.read_csv(path)

Name = []
tempArray = []
for i in range(0,21442,1072):
    data = data_set.iloc[i:i+1072]
    tempArray.append(data)
tempArray = tempArray[0:20]


index = 0
for i in tempArray:
    name = i.head(1)['GOOD_CODE'].to_frame()
    name = name.at[index,'GOOD_CODE']
    index+=1072
    Name.append(name)

data_set = data_set.drop(['STORE_CODE'],axis=1)
data_set = data_set.drop(['GOOD_CODE'],axis=1)
data_set = data_set.drop(['PDICT1'],axis=1)
data_set = data_set.drop(['BWENDU'],axis=1)
data_set = data_set.drop(['YWENDU'],axis=1)
data_set = data_set.drop(['TIANQI'],axis=1)
data_set = data_set.drop(['FENGXIANG'],axis=1)
data_set = data_set.drop(['FENGLI'],axis=1)
data_set = data_set.drop(['AQI'],axis=1)
data_set = data_set.drop(['AQIINFO'],axis=1)
data_set = data_set.drop(['AQILEVEL'],axis=1)
cols = list(data_set)
cols.insert(23,cols.pop(cols.index('PDICT3')))
cols.insert(24,cols.pop(cols.index('PDICT5')))
cols.insert(25,cols.pop(cols.index('PDICT7')))
cols.insert(26,cols.pop(cols.index('PDICT15')))
cols.insert(27,cols.pop(cols.index('PDICTMON')))
data_set = data_set.loc[:,cols]


MyArray = []
for i in range(0,21442,1072):
    data = data_set.iloc[i:i+1072]
    MyArray.append(data)
outpath1 = "C:\\Users\\yjr\\Desktop\\"
outpath3 = ".csv"

MyArray = MyArray[0:20]
index=0
for i in MyArray:
    outpath = outpath1+str(Name[index])+outpath3
    index+=1
    i['DTIME'] = pd.to_datetime(i['DTIME']);
    i = i.sort_values(by='DTIME', ascending=False)
    i = i.drop(['DTIME'],axis=1)
    i.to_csv(outpath,index=False,header=True)


