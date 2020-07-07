import  numpy as np
import pandas as pd
import random
#获取数据集
data_set = pd.read_csv("C:\\Users\\yjr\\Desktop\\1004984.csv")

data_y = data_set['PDICT7']
data_y = data_y.astype('float')
data_y = np.array(data_y)

for x in np.nditer(data_y, op_flags=['readwrite']):
    a = random.randrange(-5,5)
    if x+a < 0:
        x = random.randrange(0,5)
        continue
    x+=a

result = pd.DataFrame(data_y,columns=['0'])
result = result.astype('int')
#输出
output = 'C:\\Users\\yjr\\Desktop\\result.csv'
result.to_csv(output,index=False,header=False)
print('已输出')
