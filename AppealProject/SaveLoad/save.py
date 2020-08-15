import torch
import  torch.nn as nn
import  numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
import joblib
#获取数据集
data_set = pd.read_csv("C:\\Users\\yjr\\Desktop\\2004275_train.csv")

#反转 使数据按照日期先后顺序排列
data_set = data_set[::-1]
data_x = data_set.drop(['PDICT3'],axis=1)
data_x = data_x.drop(['PDICT5'],axis=1)
data_x = data_x.drop(['PDICT7'],axis=1)
data_x = data_x.drop(['PDICT15'],axis=1)
data_x = data_x.drop(['PDICTMON'],axis=1)
data_x = data_x.astype('float')
data_y = data_set['PDICT7']
data_y = data_y.astype('float')
data_y = np.array(data_y)


#开始分别进行归一化处理
scalar1 = MinMaxScaler(feature_range=(-1,1))
data_x = scalar1.fit_transform(data_x)

scalar2 = MinMaxScaler(feature_range=(-1,1))
data_y = scalar2.fit_transform(data_y.reshape(-1,1))
'''
scalar1 = StandardScaler()
data_x = scalar1.fit_transform(data_x)

scalar2 = StandardScaler()
data_y = scalar2.fit_transform(data_y.reshape(-1,1))
'''

seq = 10
def create_train(x,y,seq):
    list_x = []
    for i in range(len(x) - seq + 1):
        tempx = x[i:i + seq]
        tempx = torch.FloatTensor(tempx)
        tempy = y[i:i + seq]
        tempy = torch.FloatTensor(tempy)
        list_x.append((tempx,tempy))
    return list_x

train_set = create_train(data_x,data_y,seq)

#建立模型
class MyLSTM(nn.Module):
    def __init__(self,input_size = 18,hidden_layer_size=32,output_size=10):
        super(MyLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size,self.hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, hidden_layer_size),
                            torch.zeros(1, 1, hidden_layer_size))
      #  self.dropout = torch.nn.Dropout(p=0.3)
        #self.drop_rate = 0.5
    def forward(self,input_seq):
        out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
      #  out = torch.nn.functional.dropout(out,self.drop_rate)
      #  out = self.dropout(out)
        predictions = self.linear(out.view(len(input_seq), -1))
        return predictions[-1]
model = MyLSTM()
#定义损失函数和优化器
loss_function = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
optimizer = torch.optim.Adam(model.parameters())
#训练模型
epochs = 100


for i in range(epochs):
    for my_x,my_y in train_set:
        optimizer.zero_grad()

        model.hidden_cell =  (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(my_x)
        my_y = my_y.squeeze(-1)
        single_loss = loss_function(y_pred,my_y)
        single_loss.backward()
        optimizer.step()
    if i % 10 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

#输出
output = 'C:\\Users\\yjr\\Desktop\\test.txt'
state = {'net':model.state_dict(), 'MinMax1':scalar1,'MinMax2':scalar2}
torch.save(state,output)
#torch.save(model.state_dict(),output)

print('已输出')
