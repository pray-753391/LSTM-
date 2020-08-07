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
scalar111 = MinMaxScaler(feature_range=(-1,1))
scalar112 = MinMaxScaler(feature_range=(-1,1))
scalar121 = MinMaxScaler(feature_range=(-1,1))
scalar122 = MinMaxScaler(feature_range=(-1,1))
scalar131 = MinMaxScaler(feature_range=(-1,1))
scalar132 = MinMaxScaler(feature_range=(-1,1))
scalar141 = MinMaxScaler(feature_range=(-1,1))
scalar142 = MinMaxScaler(feature_range=(-1,1))

def create_data(path,scalar1,scalar2):
    # 获取数据集
    data_set = pd.read_csv(path)

    # 反转 使数据按照日期先后顺序排列
    data_set = data_set[::-1]
    data_x = data_set.drop(['PDICT3'], axis=1)
    data_x = data_x.drop(['PDICT5'], axis=1)
    data_x = data_x.drop(['PDICT7'], axis=1)
    data_x = data_x.drop(['PDICT15'], axis=1)
    data_x = data_x.drop(['PDICTMON'], axis=1)
    data_x = data_x.astype('float')
    data_y = data_set['PDICT5']
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
        tempy = data_y[i:i + seq]
        tempy = torch.FloatTensor(tempy)
        list_x.append((tempx, tempy))
    train_set = list_x
    list_x = []
    for i in range(len(data_x) - seq + 1):
        tempx = data_x[i + 1:i + 1 + seq]
        tempx = torch.FloatTensor(tempx)
        #        tempy = y[i+1:i + seq+1]
        list_x.append(tempx)
    test_set = list_x
    return train_set, test_set, data_y

train_set_1000223,test_set_1000223,data_y_1000223 = create_data("C:\\Users\\yjr\\Desktop\\1000223.csv",scalar11,scalar12)
train_set_1002188,test_set_1002188,data_y_1002188 = create_data("C:\\Users\\yjr\\Desktop\\1002188.csv",scalar21,scalar22)
train_set_1002764,test_set_1002764,data_y_1002764 = create_data("C:\\Users\\yjr\\Desktop\\1002764.csv",scalar31,scalar32)
train_set_1002822,test_set_1002822,data_y_1002822 = create_data("C:\\Users\\yjr\\Desktop\\1002822.csv",scalar41,scalar42)
train_set_1002267,test_set_1002267,data_y_1002267 = create_data("C:\\Users\\yjr\\Desktop\\1002267.csv",scalar51,scalar52)
train_set_1002579,test_set_1002579,data_y_1002579 = create_data("C:\\Users\\yjr\\Desktop\\1002579.csv",scalar61,scalar62)
train_set_1002776,test_set_1002776,data_y_1002776 = create_data("C:\\Users\\yjr\\Desktop\\1002776.csv",scalar71,scalar72)
train_set_1000016,test_set_1000016,data_y_1000016 = create_data("C:\\Users\\yjr\\Desktop\\1000016.csv",scalar81,scalar82)
train_set_1000627,test_set_1000627,data_y_1000627 = create_data("C:\\Users\\yjr\\Desktop\\1000627.csv",scalar91,scalar92)
train_set_1000138,test_set_1000138,data_y_1000138 = create_data("C:\\Users\\yjr\\Desktop\\1000138.csv",scalar101,scalar102)
train_set_1000146,test_set_1000146,data_y_1000146 = create_data("C:\\Users\\yjr\\Desktop\\1000146.csv",scalar111,scalar112)
train_set_1000321,test_set_1000321,data_y_1000321 = create_data("C:\\Users\\yjr\\Desktop\\1000321.csv",scalar121,scalar122)
train_set_1000643,test_set_1000643,data_y_1000643 = create_data("C:\\Users\\yjr\\Desktop\\1000643.csv",scalar131,scalar132)
train_set_1000363,test_set_1000363,data_y_1000363 = create_data("C:\\Users\\yjr\\Desktop\\1000363.csv",scalar141,scalar142)

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

model1 = MyLSTM()
model2 = MyLSTM()
model3 = MyLSTM()
model4 = MyLSTM()
model5 = MyLSTM()
model6 = MyLSTM()
model7 = MyLSTM()
model8 = MyLSTM()
model9 = MyLSTM()
model10 = MyLSTM()
model11 = MyLSTM()
model12 = MyLSTM()
model13 = MyLSTM()
model14 = MyLSTM()


#定义损失函数和优化器

loss_function1 = nn.MSELoss()
loss_function2 = nn.MSELoss()
loss_function3 = nn.MSELoss()
loss_function4 = nn.MSELoss()
loss_function5 = nn.MSELoss()
loss_function6 = nn.MSELoss()
loss_function7 = nn.MSELoss()
loss_function8 = nn.MSELoss()
loss_function9 = nn.MSELoss()
loss_function10 = nn.MSELoss()
loss_function11 = nn.MSELoss()
loss_function12 = nn.MSELoss()
loss_function13 = nn.MSELoss()
loss_function14 = nn.MSELoss()

optimizer1 = torch.optim.Adam(model1.parameters())
optimizer2 = torch.optim.Adam(model2.parameters())
optimizer3 = torch.optim.Adam(model3.parameters())
optimizer4 = torch.optim.Adam(model4.parameters())
optimizer5 = torch.optim.Adam(model5.parameters())
optimizer6 = torch.optim.Adam(model6.parameters())
optimizer7 = torch.optim.Adam(model7.parameters())
optimizer8 = torch.optim.Adam(model8.parameters())
optimizer9 = torch.optim.Adam(model9.parameters())
optimizer10 = torch.optim.Adam(model10.parameters())
optimizer11 = torch.optim.Adam(model11.parameters())
optimizer12 = torch.optim.Adam(model12.parameters())
optimizer13 = torch.optim.Adam(model13.parameters())
optimizer14 = torch.optim.Adam(model14.parameters())

#训练模型
epochs = 150


def train(train_set,model,optimizer,loss_function):
    for my_x,my_y in train_set:
        optimizer.zero_grad()

        model.hidden_cell =  (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(my_x)
        my_y = my_y.squeeze(-1)
        single_loss = loss_function(y_pred,my_y)
        single_loss.backward()
        optimizer.step()


def test(test_set,model,scalar2):
    result=[]
    for i in test_set:
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            a = model(i)
            a = np.array(a).reshape(-1, 1)
            actual_predictions = scalar2.inverse_transform(a)
            temp = actual_predictions[-1]
            result.append(temp[0])
    return  result



for i in range(epochs):
    train(train_set_1000223,model1,optimizer1,loss_function1)
#开始预测
model1 = model1.eval()
result_1000223 = test(test_set_1000223,model1,scalar12)
result_1000223.reverse()
result_1000223 = pd.DataFrame(result_1000223)
result_1000223=result_1000223.astype(int)
print('1')

for i in range(epochs):
    train(train_set_1002188,model2,optimizer2,loss_function2)
model2 = model2.eval()
result_1002188 = test(test_set_1002188,model2,scalar22)
result_1002188.reverse()
result_1002188 = pd.DataFrame(result_1002188)
result_1002188=result_1002188.astype(int)
print('2')
for i in range(epochs):
    train(train_set_1002764,model3,optimizer3,loss_function3)
model3 = model3.eval()
result_1002764 = test(test_set_1002764,model3,scalar32)
result_1002764.reverse()
result_1002764 = pd.DataFrame(result_1002764)
result_1002764=result_1002764.astype(int)
print('3')
for i in range(epochs):
    train(train_set_1002822,model4,optimizer4,loss_function4)
model4 = model4.eval()
result_1002822 = test(test_set_1002822,model4,scalar42)
result_1002822.reverse()
result_1002822 = pd.DataFrame(result_1002822)
result_1002822=result_1002822.astype(int)
print('4')
for i in range(epochs):
    train(train_set_1002267,model5,optimizer5,loss_function5)
model5 = model5.eval()
result_1002267 = test(test_set_1002267,model5,scalar52)
result_1002267.reverse()
result_1002267 = pd.DataFrame(result_1002267)
result_1002267=result_1002267.astype(int)
print('5')
for i in range(epochs):
    train(train_set_1002579,model6,optimizer6,loss_function6)
model6 = model6.eval()
result_1002579 = test(test_set_1002579,model6,scalar62)
result_1002579.reverse()
result_1002579 = pd.DataFrame(result_1002579)
result_1002579=result_1002579.astype(int)
print('6')

for i in range(epochs):
    train(train_set_1002776,model7,optimizer7,loss_function7)
model7 = model7.eval()
result_1002776 = test(test_set_1002776,model7,scalar72)
result_1002776.reverse()
result_1002776 = pd.DataFrame(result_1002776)
result_1002776=result_1002776.astype(int)
print('7')

for i in range(epochs):
    train(train_set_1000016,model8,optimizer8,loss_function8)
model8 = model8.eval()
result_1000016 = test(test_set_1000016,model8,scalar82)
result_1000016.reverse()
result_1000016 = pd.DataFrame(result_1000016)
result_1000016=result_1000016.astype(int)
print('8')
for i in range(epochs):
    train(train_set_1000627,model9,optimizer9,loss_function9)
model9 = model9.eval()
result_1000627 = test(test_set_1000627,model9,scalar92)
result_1000627.reverse()
result_1000627 = pd.DataFrame(result_1000627)
result_1000627=result_1000627.astype(int)
print('9')
for i in range(epochs):
    train(train_set_1000138,model10,optimizer10,loss_function10)
model10 = model10.eval()
result_1000138 = test(test_set_1000138,model10,scalar102)
result_1000138.reverse()
result_1000138 = pd.DataFrame(result_1000138)
result_1000138=result_1000138.astype(int)
print('10')
for i in range(epochs):
    train(train_set_1000146,model11,optimizer11,loss_function11)
model11 = model11.eval()
result_1000146 = test(test_set_1000146,model11,scalar112)
result_1000146.reverse()
result_1000146 = pd.DataFrame(result_1000146)
result_1000146=result_1000146.astype(int)
print('11')
for i in range(epochs):
    train(train_set_1000321,model12,optimizer12,loss_function12)
model12 = model12.eval()
result_1000321 = test(test_set_1000321,model12,scalar122)
result_1000321.reverse()
result_1000321 = pd.DataFrame(result_1000321)
result_1000321=result_1000321.astype(int)
print('12')
for i in range(epochs):
    train(train_set_1000643,model13,optimizer13,loss_function13)
model13 = model13.eval()
result_1000643 = test(test_set_1000643,model13,scalar132)
result_1000643.reverse()
result_1000643 = pd.DataFrame(result_1000643)
result_1000643=result_1000643.astype(int)
print('13')
for i in range(epochs):
    train(train_set_1000363,model14,optimizer14,loss_function14)
model14 = model14.eval()
result_1000363 = test(test_set_1000363,model14,scalar142)
result_1000363.reverse()
result_1000363 = pd.DataFrame(result_1000363)
result_1000363=result_1000363.astype(int)
print('14')

result = pd.concat([result_1002188,result_1002764,result_1002822,result_1002267,result_1002579,result_1002776],axis=1)
#输出
output = 'C:\\Users\\yjr\\Desktop\\result5.csv'
result.to_csv(output,index=False,header=False)
print('已输出')
