'''
https://zhuanlan.zhihu.com/p/105638031
'''
import torch
import  torch.nn as nn
import  seaborn as sns
import  numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
#任务是根据前132个月来预测最近12个月内旅行的乘客人数。
# 我们有144个月的记录，这意味着前132个月的数据将用于训练我们的LSTM模型，
# 而模型性能将使用最近12个月的值进行评估。

#导入数据集
data_csv = pd.read_csv("C:\\Users\\yjr\\Desktop\\AirPassengers.csv",usecols=[1])
#数据集进行预处理
all_data = data_csv.values.astype('float')
all_data = all_data.reshape(-1)

#接下来，我们将数据集分为训练集和测试集。LSTM算法将在训练集上进行训练。然后将使用该模型对测试集进行预测。
# 将预测结果与测试集中的实际值进行比较，以评估训练后模型的性能。
#前132条记录将用于训练模型，后12条记录将用作测试集。以下脚本将数据分为训练集和测试集。
test_data_size = 12
#[:-a 指取到倒数第a个为止]
train_data = all_data[:-test_data_size]
#[-a:]指从倒数第a个开始取
test_data_size = all_data[-test_data_size:]

#我们的数据集目前尚未规范化。最初几年的乘客总数远少于后来几年的乘客总数。
#标准化数据以进行时间序列预测非常重要。以在一定范围内的最小值和最大值之间对数据进行规范化。
#我们将使用模块中的MinMaxScaler类sklearn.preprocessing来扩展数据。
#以下代码 分别将最大值和最小值分别为-1和1归一化。
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
#进行归一化要求reshape为（-1，1）
train_data_normalized = scaler.fit_transform(train_data.reshape(-1,1))
#此时数据集的值在-1到1之间
#在此重要的是要提到数据标准化仅应用于训练数据，而不应用于测试数据。
# 如果对测试数据进行归一化处理，则某些信息可能会从训练集中 到测试集中。

#接着讲数据集转换为张量 换成原来的shape
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

'''
在我们的数据集中，使用12的序列长度很方便，因为我们有月度数据，一年中有12个月。
如果我们有每日数据，则更好的序列长度应该是365，即一年中的天数。因此，我们将训练的输入序列长度设置为12。

接下来，我们将定义一个名为的函数create_inout_sequences。该函数将接受原始输入数据，并将返回一个元组列表。
在每个元组中，第一个元素将包含与12个月内旅行的乘客数量相对应的12个项目的列表，
第二个元组元素将包含一个项目，即在12 + 1个月内的乘客数量。
'''
train_window = 12
def create_inout_seq(input,tw):
    inout_seq = []
    L = len(input)
    for i in range(L-tw):
        train_seq = input[i:i+tw]
        train_label = input[i+tw:i+tw+1]
        inout_seq.append((train_seq,train_label))
    return inout_seq
train_inout_seq = create_inout_seq(train_data_normalized,train_window)
#如果打印train_inout_seq列表的长度，您将看到它包含120个项目。
# 这是因为尽管训练集包含132个元素，但是序列长度为12，这意味着第一个序列由前12个项目组成，第13个项目是第一个序列的标签。
# 同样，第二个序列从第二个项目开始，到第13个项目结束，而第14个项目是第二个序列的标签，依此类推。

#开始创建模型
class MyLSTM(nn.Module):
    '''
    LSTM该类的构造函数接受三个参数：
    input_size：对应于输入中的要素数量。尽管我们的序列长度为12，但每个月我们只有1个值，即乘客总数，因此输入大小为1。
    hidden_layer_size：指定隐藏层的数量以及每层中神经元的数量。我们将有一层100个神经元。
    output_size：输出中的项目数，由于我们要预测未来1个月的乘客人数，因此输出大小为1。
    '''
    def __init__(self,input_size = 1,hidden_layer_size=4,output_size=1):
        super(MyLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        '''
        在构造函数中，我们创建变量lstm，linear，和hidden_cell。
        LSTM算法接受三个输入：先前的隐藏状态，先前的单元状态和当前输入。
        lstm和linear层变量用于创建LSTM和线性层。
        而hidden_cell变量包含先前的隐藏状态和单元状态
        '''
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size,output_size)
        self.hidden_cell = (torch.zeros(1,1,hidden_layer_size),
                            torch.zeros(1,1,hidden_layer_size))
    def forward(self,input_seq):
        '''
        在forward方法内部，将input_seq作为参数传递，该参数首先传递给lstm图层。
        lstm层的输出是当前时间步的隐藏状态和单元状态，以及输出。
        lstm图层的输出将传递到该linear图层。
        预计的乘客人数存储在predictions列表的最后一项中，并返回到调用函数。
        '''
        out,self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(out.view(len(input_seq), -1))
        return predictions[-1]
model = MyLSTM()
#定义损失函数和优化器
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
#训练模型
epochs = 500
for i in range(epochs):
    for seq,labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell =  (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        single_loss = loss_function(y_pred,labels)
        single_loss.backward()
        optimizer.step()
    if i % 100 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

#模型以及训练完毕，开始预测
fut_pred = 12
#该test_inputs项目将包含12个项目。
test_inputs = train_data_normalized[-train_window:].tolist()
model.eval()
# 在for循环内，这12个项目将用于对测试集中的第一个项目进行预测，即项目编号133。
# 然后将预测值附加到test_inputs列表中。
# 在第二次迭代中，最后12个项目将再次用作输入，并将进行新的预测，然后将其test_inputs再次添加到列表中。
# for由于测试集中有12个元素，因此该循环将执行12次。
# 在循环末尾，testvinputs列表将包含24个项目。最后12个项目将是测试集的预测值。
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())
#由于我们对训练数据集进行了标准化，因此预测值也进行了标准化。我们需要将归一化的预测值转换为实际的预测值。
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

#绘图 查看区别
actual_predictions = actual_predictions.reshape(-1)
real = data_csv.values
real = real[-12:]
plt.plot(actual_predictions,'r',label = 'prediction')
plt.plot(real,'b',label='real')
plt.show()
