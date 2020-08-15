import torch
import  torch.nn as nn
import  numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
#获取数据集
data_set = pd.read_csv("C:\\Users\\yjr\\Desktop\\2004275_test.csv")
data_load = torch.load('C:\\Users\\yjr\\Desktop\\test.txt')
scalar1 = data_load['MinMax1']
scalar2 = data_load['MinMax2']
#反转 使数据按照日期先后顺序排列
data_set = data_set[::-1]
data_x = data_set.astype('float')

data_x = scalar1.transform(data_x)


seq=10
def create_test(x,seq):
    list_x = []
    for i in range(len(x) - seq ):
        tempx = x[i:i+seq]
        tempx = torch.FloatTensor(tempx)
        list_x.append(tempx)
    return list_x

test_set = create_test(data_x,seq)

#建立模型
class MyLSTM(nn.Module):
    def __init__(self,input_size = 18,hidden_layer_size=32,output_size=10):
        super(MyLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size,self.hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, hidden_layer_size),
                            torch.zeros(1, 1, hidden_layer_size))

    def forward(self,input_seq):
        out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(out.view(len(input_seq), -1))
        return predictions[-1]
model = MyLSTM()
model.load_state_dict(data_load['net'])


#开始预测
model = model.eval()
result = []
for i in test_set:
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        a = model(i)
        a = np.array(a).reshape(-1, 1)
        actual_predictions = scalar2.inverse_transform(a)
        temp = actual_predictions[-1]
        result.append(temp[0])

#把result导出
#先翻转
result.reverse()

result = np.array(result)
data = []
for x in np.nditer(result, op_flags=['readwrite']):
    if x < 0:
        x = 0
    data.append(x)
result = pd.DataFrame(data,columns=['MYPDICT7'])
#result = result[1:]
result = result.astype('int')

#输出
output = 'C:\\Users\\yjr\\Desktop\\result7.csv'


#out_put_data.to_csv(output,index=False,header=True,mode = 'a')
result.to_csv(output,index=False,header=True)
print('已输出')
