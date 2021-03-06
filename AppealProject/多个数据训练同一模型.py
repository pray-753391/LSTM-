import torch
import  torch.nn as nn
import  numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scalar11 = MinMaxScaler(feature_range=(-1,1))
scalar12 = MinMaxScaler(feature_range=(-1,1))
scalar21 = MinMaxScaler(feature_range=(-1,1))
scalar22 = MinMaxScaler(feature_range=(-1,1))
scalar31 = MinMaxScaler(feature_range=(-1,1))
scalar32 = MinMaxScaler(feature_range=(-1,1))
scalar41 = MinMaxScaler(feature_range=(-1,1))
scalar42 = MinMaxScaler(feature_range=(-1,1))
scalar51 = MinMaxScaler(feature_range=(-1,1))
scalar52 = MinMaxScaler(feature_range=(-1,1))
scalar61 = MinMaxScaler(feature_range=(-1,1))
scalar62 = MinMaxScaler(feature_range=(-1,1))
scalar71 = MinMaxScaler(feature_range=(-1,1))
scalar72 = MinMaxScaler(feature_range=(-1,1))
scalar81 = MinMaxScaler(feature_range=(-1,1))
scalar82 = MinMaxScaler(feature_range=(-1,1))
scalar91 = MinMaxScaler(feature_range=(-1,1))
scalar92 = MinMaxScaler(feature_range=(-1,1))
scalar101 = MinMaxScaler(feature_range=(-1,1))
scalar102 = MinMaxScaler(feature_range=(-1,1))
def create_data(path,scalar1,scalar2):
    # 获取数据集
    data_set = pd.read_csv(path)
    data_set = data_set.drop(['DTIME'], axis=1)
    data_set = data_set.drop(['GOOD_CODE'], axis=1)
    # 反转 使数据按照日期先后顺序排列
    data_set = data_set[::-1]
    data_x = data_set.drop(['PDICT3'], axis=1)
    data_x = data_x.drop(['PDICT5'], axis=1)
    data_x = data_x.drop(['PDICT7'], axis=1)
    data_x = data_x.astype('float')
    data_y = data_set['PDICT3']
    data_y = data_y.astype('float')
    data_y = np.array(data_y)

    # 开始分别进行归一化处理
    data_x = scalar1.fit_transform(data_x)
    data_y = scalar2.fit_transform(data_y.reshape(-1, 1))


    seq = 10
    list_x = []
    for i in range(len(data_x) - seq + 1):
        tempx = data_x[i:i + seq]
        tempx = torch.FloatTensor(tempx)
        tempy = data_y[i:i+seq]
        tempy = torch.FloatTensor(tempy)
        list_x.append((tempx,tempy))
    train_set = list_x
    list_x = []
    for i in range(len(data_x) - seq +1):
        tempx = data_x[i+1:i+1+seq]
        tempx = torch.FloatTensor(tempx)
        list_x.append(tempx)
    test_set = list_x
    return train_set,test_set,data_y

train_set_5006044,test_set_5006044,data_y_5006044 = create_data("C:\\Users\\yjr\\Desktop\\5006044.csv",scalar11,scalar12)
train_set_1013260,test_set_1013260,data_y_1013260 = create_data("C:\\Users\\yjr\\Desktop\\1013260.csv",scalar21,scalar22)
train_set_1013153,test_set_1013153,data_y_1013153 = create_data("C:\\Users\\yjr\\Desktop\\1013153.csv",scalar31,scalar32)
train_set_1016924,test_set_1016924,data_y_1016924 = create_data("C:\\Users\\yjr\\Desktop\\1016924.csv",scalar41,scalar42)
train_set_1003975,test_set_1003975,data_y_1003975 = create_data("C:\\Users\\yjr\\Desktop\\1003975.csv",scalar51,scalar52)
train_set_1000346,test_set_1000346,data_y_1000346 = create_data("C:\\Users\\yjr\\Desktop\\1000346.csv",scalar61,scalar62)
train_set_1007265,test_set_1007265,data_y_1007265 = create_data("C:\\Users\\yjr\\Desktop\\1007265.csv",scalar71,scalar72)
train_set_1001608,test_set_1001608,data_y_1001608 = create_data("C:\\Users\\yjr\\Desktop\\1001608.csv",scalar81,scalar82)
train_set_1004984,test_set_1004984,data_y_1004984 = create_data("C:\\Users\\yjr\\Desktop\\1004984.csv",scalar91,scalar92)
train_set_5005122,test_set_5005122,data_y_5005122 = create_data("C:\\Users\\yjr\\Desktop\\5005122.csv",scalar101,scalar102)



#建立模型
class MyLSTM(nn.Module):
    def __init__(self,input_size = 11,hidden_layer_size=128,output_size=10):
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
epochs = 100


def train(train_set):
    for my_x,my_y in train_set:
        optimizer.zero_grad()
        model.hidden_cell =  (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(my_x)
        my_y = my_y.squeeze(-1)
        single_loss = loss_function(y_pred,my_y)
        single_loss.backward()
        optimizer.step()


for i in range(epochs):
    train(train_set_5006044)
    train(train_set_1013260)
    train(train_set_1013153)
    train(train_set_1016924)
    train(train_set_1003975)
    train(train_set_1000346)
    train(train_set_1007265)
    train(train_set_1001608)
    train(train_set_1004984)
    train(train_set_5005122)
    if i % 10 == 0:
        print(f'epoch: {i:3}')
        #print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')


#开始预测
model = model.eval()
def test(test_set,scalar2):
    result=[]
    for i in test_set:
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),torch.zeros(1, 1, model.hidden_layer_size))
            a = model(i)
            a = np.array(a).reshape(-1, 1)
            actual_predictions = scalar2.inverse_transform(a)
            temp = actual_predictions[-1]
            result.append(temp[0])
    return result

result_5006044 = test(test_set_5006044,scalar12)
result_5006044.reverse()
result_5006044 = pd.DataFrame(result_5006044)
result_5006044=result_5006044.astype(int)

result_1013260 = test(test_set_1013260,scalar22)
result_1013260.reverse()
result_1013260 = pd.DataFrame(result_1013260)
result_1013260=result_1013260.astype(int)

result_1013153 = test(test_set_1013153,scalar32)
result_1013153.reverse()
result_1013153 = pd.DataFrame(result_1013153)
result_1013153=result_1013153.astype(int)

result_1016924 = test(test_set_1016924,scalar42)
result_1016924.reverse()
result_1016924 = pd.DataFrame(result_1016924)
result_1016924=result_1016924.astype(int)

result_1003975 = test(test_set_1003975,scalar52)
result_1003975.reverse()
result_1003975 = pd.DataFrame(result_1003975)
result_1003975=result_1003975.astype(int)

result_1000346 = test(test_set_1000346,scalar62)
result_1000346.reverse()
result_1000346 = pd.DataFrame(result_1000346)
result_1000346=result_1000346.astype(int)

result_1007265 = test(test_set_1007265,scalar72)
result_1007265.reverse()
result_1007265 = pd.DataFrame(result_1007265)
result_1007265=result_1007265.astype(int)

result_1001608 = test(test_set_1001608,scalar82)
result_1001608.reverse()
result_1001608 = pd.DataFrame(result_1001608)
result_1001608=result_1001608.astype(int)

result_1004984 = test(test_set_1004984,scalar92)
result_1004984.reverse()
result_1004984 = pd.DataFrame(result_1004984)
result_1004984=result_1004984.astype(int)

result_5005122 = test(test_set_5005122,scalar102)
result_5005122.reverse()
result_5005122 = pd.DataFrame(result_5005122)
result_5005122=result_5005122.astype(int)

result = pd.concat([result_5006044,result_1013260,result_1013153,result_1016924,result_1003975,result_1000346,result_1007265,result_1001608,result_1004984,result_5005122],axis=1)
 #1000346 1004984
#输出
output = 'C:\\Users\\yjr\\Desktop\\result.csv'
result.to_csv(output,index=False,header=False)
print('已输出')
