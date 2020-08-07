import  numpy as np
import pandas as pd
import random
#获取数据集
data_set = pd.read_csv("C:\\Users\\yjr\\Desktop\\1000542.csv")
data_3 = data_set['PDICT3']
data_5= data_set['PDICT5']
data_7 = data_set['PDICT7']
data_15 = data_set['PDICT15']
data_MON = data_set['PDICTMON']

data_3 = data_3.astype('float')
data_5 = data_5.astype('float')
data_7 = data_7.astype('float')
data_15 = data_15.astype('float')
data_MON = data_MON.astype('float')

data_3 = np.array(data_3)
data_5 = np.array(data_5)
data_7 = np.array(data_7)
data_15 = np.array(data_15)
data_MON = np.array(data_MON)

data = []
for x in np.nditer(data_3, op_flags=['readwrite']):
    a = random.randrange(-3, 3)
    if x > 0:
        while x+a < 0:
            print(x+a)
            a = random.randrange(-3, 3)
        y = x+a
        data.append(y)
    else:
        data.append(x)
result_3 = pd.DataFrame(data,columns=['MYPDICT3'])
result_3 = result_3.astype('int')

data = []
for x in np.nditer(data_5, op_flags=['readwrite']):
    a = random.randrange(-3, 3)
    if x > 0:
        while x+a < 0:
            print(x+a)
            a = random.randrange(-3, 3)
        y = x+a
        data.append(y)
    else:
        data.append(x)
result_5 = pd.DataFrame(data,columns=['MYPDICT5'])
result_5 = result_5.astype('int')

data = []
for x in np.nditer(data_7, op_flags=['readwrite']):
    a = random.randrange(-3, 3)
    if x > 0:
        while x+a < 0:
            print(x+a)
            a = random.randrange(-3, 3)
        y = x+a
        data.append(y)
    else:
        data.append(x)
result_7 = pd.DataFrame(data,columns=['MYPDICT7'])
result_7 = result_7.astype('int')

data = []
for x in np.nditer(data_15, op_flags=['readwrite']):
    a = random.randrange(-3, 3)
    if x > 0:
        while x+a < 0:
            print(x+a)
            a = random.randrange(-3, 3)
        y = x+a
        data.append(y)
    else:
        data.append(x)
result_15 = pd.DataFrame(data,columns=['MYPDICT15'])
result_15 = result_15.astype('int')

data = []
for x in np.nditer(data_MON, op_flags=['readwrite']):
    a = random.randrange(-3, 3)
    if x > 0:
        while x+a < 0:
            print(x+a)
            a = random.randrange(-3, 3)
        y = x+a
        data.append(y)
    else:
        data.append(x)
result_MON = pd.DataFrame(data,columns=['MYPDICTMON'])
result_MON = result_MON.astype('int')

result = pd.concat([result_3,result_5,result_7,result_15,result_MON],axis=1)

#输出
output = 'C:\\Users\\yjr\\Desktop\\test.csv'
result.to_csv(output,index=False,header=True)
print('已输出')
