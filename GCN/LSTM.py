'''
参考链接：https://blog.csdn.net/ch206265/article/details/106962354?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-6.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-6.control
'''
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import pandas as pd
"""
导入数据
"""
data=pd.read_csv(r"C:\Users\13824\OneDrive\学习记录\汇总\总结与记录\project\STARMA&GCN\STARMA-GCN\STARMA\test_data.csv")
coordinate=pd.read_csv(r"C:\Users\13824\OneDrive\学习记录\汇总\总结与记录\project\STARMA&GCN\STARMA-GCN\STARMA\test_data(coordinate).csv")
"""
数据预处理
"""
data.columns#显示数据集中 列的数据类型
all_data = data['V71'].values.astype(float)#将passengers列的数据类型改为float
all_data=all_data[0:720]
#划分测试集和训练集
test_data_size = 12
train_data = all_data[:-test_data_size]#除了最后144个数据，其他全取
test_data = all_data[-test_data_size:]#取最后144个数据
print(len(train_data))
print(len(test_data))

#最大最小缩放器进行归一化，减小误差，注意，数据标准化只应用于训练数据而不应用于测试数据
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(-1, 1))
# train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
#查看归一化之后的前5条数据和后5条数据
# print(train_data_normalized[:5])
# print(train_data_normalized[-5:])
#将数据集转换为tensor，因为PyTorch模型是使用tensor进行训练的，并将训练数据转换为输入序列和相应的标签
# train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_data=torch.FloatTensor(train_data)
# train_data_normalized.shape
#view相当于numpy中的resize,参数代表数组不同维的维度；
#参数为-1表示，这个维的维度由机器自行推断，如果没有-1，那么view中的所有参数就要和tensor中的元素总个数一致

#定义create_inout_sequences函数，接收原始输入数据，并返回一个元组列表。
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]#预测time_step之后的第一个数值
        inout_seq.append((train_seq, train_label))#inout_seq内的数据不断更新，但是总量只有tw+1个
    return inout_seq
train_window = 12#设置训练输入的序列长度为12，类似于time_step = 12
train_inout_seq = create_inout_sequences(train_data, train_window)
print(train_inout_seq[:5])#产看数据集改造结果
"""
注意：
create_inout_sequences返回的元组列表由一个个序列组成，
每一个序列有13个数据，分别是设置的12个数据（train_window）+ 第13个数据（label）
第一个序列由前12个数据组成，第13个数据是第一个序列的标签。
同样，第二个序列从第二个数据开始，到第13个数据结束，而第14个数据是第二个序列的标签，依此类推。
"""

"""
创建LSTM模型
参数说明：
1、input_size:对应的及特征数量，此案例中为1，即passengers
2、output_size:预测变量的个数，及数据标签的个数
2、hidden_layer_size:隐藏层的特征数，也就是隐藏层的神经元个数
"""
class LSTM(nn.Module):#注意Module首字母需要大写
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # 创建LSTM层和linear层，LSTM层提取特征，linear层用作最后的预测
        ##LSTM算法接受三个输入：先前的隐藏状态，先前的单元状态和当前输入。
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        #初始化隐含状态及细胞状态C，hidden_cell变量包含先前的隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),  #h0和C0
                            torch.zeros(1, 1, self.hidden_layer_size))
                            # 为什么的第二个参数也是1
                            # 第二个参数代表的应该是batch_size吧
                            # 是因为之前对数据已经进行过切分了吗？？？？？

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        #lstm的输出是当前时间步的隐藏状态ht和单元状态ct以及输出lstm_out
        #按照lstm的格式修改input_seq的形状，作为linear层的输入
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]#返回predictions的最后一个元素
"""
forward方法：LSTM层的输入与输出：out, (ht,Ct)=lstm(input,(h0,C0)),其中
一、输入格式：lstm(input,(h0, C0))
1、input为（seq_len,batch,input_size）格式的tensor,seq_len即为time_step
2、h0为(num_layers * num_directions, batch, hidden_size)格式的tensor，隐藏状态的初始状态
3、C0为(seq_len, batch, input_size）格式的tensor，细胞初始状态
二、输出格式：output,(ht,Ct)
1、output为(seq_len, batch, num_directions*hidden_size）格式的tensor，包含输出特征h_t(源于LSTM每个t的最后一层)
2、ht为(num_layers * num_directions, batch, hidden_size)格式的tensor，
3、Ct为(num_layers * num_directions, batch, hidden_size)格式的tensor，
"""

#创建LSTM()类的对象，定义损失函数和优化器
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#建立优化器实例
print(model)

for seq, labels in train_inout_seq:
    print(seq.shape)
    print(labels.shape)

"""
模型训练
batch-size是指1次迭代所使用的样本量；
epoch是指把所有训练数据完整的过一遍；
由于默认情况下权重是在PyTorch神经网络中随机初始化的，因此可能会获得不同的值。
"""
epochs = 150
for i in range(epochs):
    for seq, labels in train_inout_seq:
        #清除网络先前的梯度值
        optimizer.zero_grad()#训练模型时需要使模型处于训练模式，即调用model.train()。缺省情况下梯度是累加的，需要手工把梯度初始化或者清零，调用optimizer.zero_grad()
        #初始化隐藏层数据
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
        #实例化模型
        y_pred = model(seq)
        #计算损失，反向传播梯度以及更新模型参数
        single_loss = loss_function(y_pred, labels)#训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
        single_loss.backward()#调用loss.backward()自动生成梯度，
        optimizer.step()#使用optimizer.step()执行优化器，把梯度传播回每个网络

    # 查看模型训练的结果
    if i%25 == 1:
        print(f'epoch:{i:3} loss:{single_loss.item():10.8f}')
print(f'epoch:{i:3} loss:{single_loss.item():10.10f}')

torch.save(model,'LSTM.pkl')

"""
预测
注意，test_input中包含12个数据，
在for循环中，12个数据将用于对测试集的第一个数据进行预测，然后将预测值附加到test_inputs列表中。
在第二次迭代中，最后12个数据将再次用作输入，并进行新的预测，然后 将第二次预测的新值再次添加到列表中。
由于测试集中有12个元素，因此该循环将执行12次。
循环结束后，test_inputs列表将包含24个数据，其中，最后12个数据将是测试集的预测值。
"""
fut_pred = 12
test_inputs = all_data[-train_window:].tolist()#首先打印出数据列表的最后12个值
print(test_inputs)

#更改模型为测试或者验证模式
model.eval()#把training属性设置为false,使模型处于测试或验证状态
pred_list=[]
for i in range(12):
    data_for_pred=all_data[:(-12+i)]
    pred_list.append(data_for_pred[(-12):])

pred=[]
for i in range(12):
    pred.append(model(torch.FloatTensor(pred_list[i])).item())

pred
test_data.tolist()
np.mean(np.array(pred)-test_data)
# for i in range(fut_pred):
#     seq = torch.FloatTensor(test_inputs[-train_window:])
#     with torch.no_grad():
#         model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
#                         torch.zeros(1, 1, model.hidden_layer_size))
#         test_inputs.append(model(seq).item())
# #打印最后的12个预测值
print(test_inputs[fut_pred:])
#由于对训练集数据进行了标准化，因此预测数据也是标准化了的
#需要将归一化的预测值转换为实际的预测值。通过inverse_transform实现
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
print(actual_predictions)
print(data.values[-train_window:,1:2])
"""
根据实际值，绘制预测值
"""
x = np.arange(132, 132+fut_pred, 1)
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(data.values[-train_window:,1])
plt.plot(x, actual_predictions)
plt.show()
#绘制最近12个月的实际和预测乘客数量,以更大的尺度观测数据
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(data['V71'][-train_window:])
plt.plot(x, actual_predictions)
plt.show()