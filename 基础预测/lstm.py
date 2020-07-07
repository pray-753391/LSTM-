


#使用LSTM预测航班
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\yjr\\Desktop\\AirPassengers.csv", usecols=[1])
plt.plot(data)

#对数据进行预处理
#删除缺失的数据
data = data.dropna()
#创立数据集
dataset = data.values
#改变数据集中数值类型
dataset = dataset.astype('float32')
'''

归一化是将各个特征量化到统一的区间。 数据标准化（归一化）处理是数据挖掘的一项基础工作，
不同评价指标往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果，
为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性。
原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价。
可参考网址：https://www.cnblogs.com/bjwu/p/8977141.html
https://blog.csdn.net/coco_1998_2/article/details/83623559
'''

#开始归一化
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value-min_value
#map会根据提供的函数对指定序列做映射。
#第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
#lambda_x：匿名函数 x/scalar:函数具体执行操作 dataset:x数值的来源
#要输出map结果要加类型转换list()
dataset = list(map(lambda  x: x/scalar,dataset))
'''

look_back 就是预测下一步所需要的 time steps,
timesteps 就是 LSTM 认为每个输入数据与前多少个陆续输入的数据有联系。
time_step就是说要指定多长的序列能够构成一个上下文相关的序列。
例如诗歌，time steps很明确就是一首诗的长度。
例如有这样一段序列数据 “…ABCDBCEDF…”，
当 timesteps 为 3 时，在模型预测中如果输入数据为“D”，
那么之前接收的数据如果为“B”和“C”则此时的预测输出为 B 的概率更大，
之前接收的数据如果为“C”和“E”，则此时的预测输出为 F 的概率更大。
直观理解可看该网址：
https://blog.csdn.net/lnaruto1234/article/details/99672601?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase
'''

#look_back=2指用第1,2个来预测第3个这种
def create_dataset(dataset,look_back=2):
    dataX,dataY = [],[]
    for i in range(len(dataset) - look_back):
        #[a:b] 从第a个取到b-1个
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return  np.array(dataX),np.array(dataY)

#x为训练集 y为预测集
data_X,data_Y = create_dataset(dataset)
#取数据集的前70%作为训练集，后30%作为预测集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size

train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]


#开始建立lstm模型的前期准备
import  torch
train_X = train_X.reshape(-1,1,2)
train_Y = train_Y.reshape(-1,1,1)
test_X = test_X.reshape(-1,1,2)
#转成torch模型
train_X = torch.from_numpy(train_X)
train_Y = torch.from_numpy(train_Y)
test_X = torch.from_numpy(test_X)

#建立LSTM模型
from torch import nn
from  torch.autograd import  Variable

#使用lstm模型时都要继承这个类
class MyLSTM(nn.Module):
    #input_size即输入的隐层维度，特征向量的长度
    #hidden_size是网络的隐藏单元个数。这个维数值是自定义的，根据根据具体业务需要决定
    #num_layers  LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果,最后输出第二个计算出的结果
    def __init__(self,input_size = 2,hidden_size = 4,output_size = 1,num_layer = 2):
        super(MyLSTM, self).__init__()
        #LSTM神经网络
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer)
        #全连接层 没搞懂作用
        self.layer2 = nn.Linear(hidden_size,output_size)
    #也没搞懂做这些有啥用
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        #view函数作用就是改变tensor的形状的
        x = x.view(s*b,h)

        x = self.layer2(x)
        x = x.view(s,b,-1)
        return  x
model = MyLSTM(2,4,1,2)

#建立损失函数和优化器
criterion = nn.MSELoss()
#optimizer是一个优化器 能够根据计算得到的梯度来更新参数。
#lr为学习率
#具体可看https://blog.csdn.net/kgzhang/article/details/77479737
optimizer = torch.optim.Adam(model.parameters(model.parameters()), lr=1e-2)

#模型开始训练
for i in range(1000):
    #将tensor转化成variable
    #因为pytorch中tensor(张量)只能放在CPU上运算，
    # 而(variable)变量是可以只用GPU进行加速计算的。
    var_x = Variable(train_X)
    var_y = Variable(train_Y)
    #前向传播
    out = model(var_x)
    #计算损失度
    loss = criterion(out,var_y)

    #以下三个步骤原因为：https://blog.csdn.net/scut_salmon/article/details/82414730

    # Step 1. 请记住 Pytorch 会累加梯度
    # 每次训练前需要清空梯度值
    optimizer.zero_grad()
    # 反向传播计算梯度
    loss.backward()
    # 更新所有参数
    optimizer.step()

  #  if (i+1) % 100 == 0: #每100次输出结果
   #     print('Epoch: {}, Loss: {:.5f}'.format(i + 1, loss.item()))
#使用模型开始进行预测
model = model.eval() #变成测试模式
#对所有数据进行一次预测
data_X = data_X.reshape(-1,1,2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = model(var_data)
#改变输出格式
pred_test = pred_test.view(-1).data.numpy()
print(pred_test)
#将预测的序列可视化
plt.plot(pred_test,'r',label = 'prediction')
plt.plot(dataset,'b',label='real')
plt.legend(loc = 'best')
#plt.show()
