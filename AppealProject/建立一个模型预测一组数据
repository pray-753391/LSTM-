import torch
import  torch.nn as nn
import  numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#获取数据集
data_set = pd.read_csv("C:\\Users\\yjr\\Desktop\\1004984.csv")
data_set = data_set.drop(['DTIME'], axis=1)
data_set = data_set.drop(['GOOD_CODE'], axis=1)
#反转 使数据按照日期先后顺序排列
data_set = data_set[::-1]
data_x = data_set.drop(['PDICT3'],axis=1)
data_x = data_x.drop(['PDICT5'],axis=1)
data_x = data_x.drop(['PDICT7'],axis=1)
data_x = data_x.astype('float')
data_y = data_set['PDICT3']
data_y = data_y.astype('float')
data_y = np.array(data_y)

#开始分别进行归一化处理
scalar1 = MinMaxScaler(feature_range=(-1,1))
data_x = scalar1.fit_transform(data_x)

scalar2 = MinMaxScaler(feature_range=(-1,1))
data_y = scalar2.fit_transform(data_y.reshape(-1,1))
'''

#对所有数据进行归一化处理
scalar = MinMaxScaler(feature_range=(-1,1))
data_set = scalar.fit_transform(data_set)
data_set = pd.DataFrame(data_set)


#提取出x
data_x = data_set.drop(columns=11)
data_x = data_x.astype('float')
data_x = np.array(data_x)
#提取出y
data_y = data_set.drop(columns=range(11))
data_y = data_y.astype('float')
data_y = np.array(data_y)

print(data_y)
'''

seq = 10
def create_train(x,y,seq):
    list_x = []
    for i in range(len(x) - seq + 1):
        tempx = x[i:i + seq]
        tempx = torch.FloatTensor(tempx)
        tempy = y[i:i+seq]
        tempy = torch.FloatTensor(tempy)
        list_x.append((tempx,tempy))
    return list_x
def create_test(x,y,seq):
    list_x = []
    for i in range(len(x) - seq +1):
        tempx = x[i+1:i+1+seq]
        tempx = torch.FloatTensor(tempx)
#        tempy = y[i+1:i + seq+1]
        list_x.append(tempx)
    return list_x
train_set = create_train(data_x,data_y,seq)
test_set = create_test(data_x,data_y,seq)

#train_x = create_train(data_x,seq)
#train_x = np.array(train_x)#此时train_x的shape为(31,10,11)
#train_y = create_train(data_y,seq)
#train_y = np.array(train_y) #此时train_y的shape为(31,10,1)


#接下来把训练的数据集转换成张量
#train_x = torch.FloatTensor(train_x)

#建立模型
class MyLSTM(nn.Module):
    def __init__(self,input_size = 11,hidden_layer_size=256,output_size=10):
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
#定义损失函数和优化器
loss_function = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
optimizer = torch.optim.Adam(model.parameters())
#训练模型
epochs = 50



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
''''''
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
#把data_y转回来 画图做比较
data_y = scalar2.inverse_transform(data_y)
data_y = data_y[len(data_y)-len(result)+1:]
data_y = data_y.reshape(-1)
plt.plot(result,'r',label = 'prediction')
plt.plot(data_y,'b',label='real')
plt.show()
#把result导出
#先翻转
result.reverse()
result = pd.DataFrame(result,columns=['0'])
result = result.astype('int')
#输出
output = 'C:\\Users\\yjr\\Desktop\\result.csv'
result.to_csv(output,index=False,header=False)
print('已输出')

