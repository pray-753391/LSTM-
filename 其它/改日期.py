import numpy as np
import pandas as pd
file_name = '2004085'
filepath1 = "C:\\Users\\yjr\\Desktop\\"
filepath3 = ".csv"
file = filepath1+file_name+filepath3
data_set = pd.read_csv(file,encoding = 'gb2312')

data = data_set['DTIME']
data = data.astype('str')
data = np.array(data)

NewArray = []

for x in data:

    split = x.split('/')
    print(split)
    tail = split[2][0:4]
    year_month_day = tail+'/'+split[1]+'/'+split[0]
    year_month_day_time = year_month_day + split[2][4:]
    NewArray.append(year_month_day_time)



NewArray = pd.DataFrame(NewArray,columns=['DTIME'])
NewArray = NewArray.astype('str')
data_set['DTIME'] = NewArray['DTIME']


output = filepath1+file_name+filepath3
data_set.to_csv(output,index=False)
print('已输出')

