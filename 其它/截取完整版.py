import numpy as np
import pandas as pd
import datetime
path = "C:\\Users\\yjr\\Desktop\\temproot.csv"
index = 0
data_set = pd.read_csv(path)

MyArray = []
for i in range(0,21442,1072):
    data = data_set.iloc[i:i+1072]
    MyArray.append(data)
outpath1 = "C:\\Users\\yjr\\Desktop\\"
outpath3 = ".csv"

MyArray = MyArray[0:20]
index=0
Rightformat = '%Y/%m/%d'

for i in MyArray:
    NewArray = []
    name = i.head(1)['GOOD_CODE'].to_frame()
    name = name.at[index,'GOOD_CODE']
    name = name.astype(int)
    i.loc[:,'DTIME'] = pd.to_datetime(i.loc[:,'DTIME']);
    i = i.sort_values(by='DTIME',ascending=False)
    outpath = outpath1+str(name)+outpath3
    i.to_csv(outpath,index=False,header=True)
    index+=1072
    print(name)
