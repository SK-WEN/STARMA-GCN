#https://blog.csdn.net/sdu_hao/article/details/104478492
#https://www.cnblogs.com/liyinggang/p/13391625.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dgl.nn.pytorch import edge_softmax, GATConv


class GATLayer(nn.Module):
    def __init__(self, g, in_dim , out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2*out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')  #根据给定的函数给出增益值
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  #拼接
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self,edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1) # 归一化每一条入边的注意力系数
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h':h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z # 每个节点的特征
        self.g.apply_edges(self.edge_attention) # 为每一条边获得其注意力系数
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim , out_dim , num_heads=1, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge


    def forward(self, h):
        head_out = [attn_head(h) for attn_head in self.heads]
        if self.merge=='cat':
            return torch.cat(head_out, dim=1)
        else:
            return torch.mean(torch.stack(head_out))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim , out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g , in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim*num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h

class LSTM(nn.Module):#注意Module首字母需要大写
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # 创建LSTM层和linear层，LSTM层提取特征，linear层用作最后的预测
        # LSTM算法接受三个输入：先前的隐藏状态，先前的单元状态和当前输入。
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        #初始化隐含状态及细胞状态C，hidden_cell变量包含先前的隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))
                            # 为什么的第二个参数也是1
                            # 第二个参数代表的应该是batch_size吧
                            # 是因为之前对数据已经进行过切分了吗？？？？？

    def forward(self, input_seq):
    	#lstm的输出是当前时间步的隐藏状态ht和单元状态ct以及输出lstm_out
        # print('test10')
        # print(input_seq.view(len(input_seq), 1, -1).float())
        # print(self.hidden_cell[0])
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1).float(), self.hidden_cell)
        # print('test11')
        #按照lstm的格式修改input_seq的形状，作为linear层的输入
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        #返回predictions的最后一个元素
        return predictions[-1]

class GAT_LSTM(nn.Module):
    def __init__(self,g, in_dim, hidden_dim , out_dim, num_heads,input_size=1, hidden_layer_size=100, output_size=1):
        super(GAT_LSTM, self).__init__()
        self.gat=GAT(g, in_dim, hidden_dim , out_dim, num_heads)
        self.lstm=LSTM(input_size=input_size, hidden_layer_size=hidden_layer_size, output_size=output_size)
        self.g=g

    def GAT_preprocess(self,data):
        # print('test7')
        buf=np.zeros(shape=(data.shape[0],data.shape[1]))
        # print('test8')
        buf=torch.tensor(buf)
        # print('test5')
        for i in range(data.shape[0]):
            # print('test6')
            h=torch.FloatTensor(data[i,])
            # print(h)
            h=self.gat(h.view(-1,1))
            buf[i,]=h.view(-1)
        return buf

    def forward(self,data):
        # print('test4')
        output_GAT=self.GAT_preprocess(data)
        # print('test9')
        # print(output_GAT[:, 0])
        # input_LSTM=torch.FloatTensor(output_GAT[:,0:1])

        return(self.lstm(output_GAT[:, 0]))

graph=pd.read_csv(r"C:\Users\13824\OneDrive\学习记录\汇总\总结与记录\project\STARMA&GCN\STARMA-GCN\STARMA\adj.csv")
graph=graph.values
graph=graph[:,1:]
src=[]
dst=[]
for i in range(graph.shape[0]):
    for j in range(graph.shape[0]):
        if graph[i,j] > 0:
            src.append(i)
            dst.append(j)
g=dgl.graph((src,dst))
g

"""
读取数据
"""
data=pd.read_csv(r"C:\Users\13824\OneDrive\学习记录\汇总\总结与记录\project\STARMA&GCN\STARMA-GCN\STARMA\test_data.csv")
coordinate=pd.read_csv(r"C:\Users\13824\OneDrive\学习记录\汇总\总结与记录\project\STARMA&GCN\STARMA-GCN\STARMA\test_data(coordinate).csv")
data.columns#显示数据集中 列的数据类型

"""
数据预处理
"""
all_data = torch.FloatTensor(data.values)#将passengers列的数据类型改为float
all_data=all_data[0:720,].numpy()
all_data=all_data[:,1:]
all_data.shape
#划分测试集和训练集
test_data_size = 12
train_data = all_data[:-test_data_size,]#除了最后12个数据，其他全取
test_data = all_data[-test_data_size:,]#取最后12个数据
print(train_data.shape)
print(test_data.shape)

#最大最小缩放器进行归一化，减小误差，注意，数据标准化只应用于训练数据而不应用于测试数据
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(-1, 1))
# train_data_normalized=train_data
# for i in range(train_data.shape[1]):
#     train_data_normalized[:,i:(i+1)] = scaler.fit_transform(train_data[:,i:(i+1)])
#查看归一化之后的前5条数据和后5条数据
# print(train_data_normalized[:5,])
# print(train_data_normalized[-5:,])
#将数据集转换为tensor，因为PyTorch模型是使用tensor进行训练的，并将训练数据转换为输入序列和相应的标签
# train_data_normalized = torch.FloatTensor(train_data_normalized)
# train_data_normalized.shape
#view相当于numpy中的resize,参数代表数组不同维的维度；
#参数为-1表示，这个维的维度由机器自行推断，如果没有-1，那么view中的所有参数就要和tensor中的元素总个数一致

#定义create_inout_sequences函数，接收原始输入数据，并返回一个元组列表。
train_data=torch.FloatTensor(train_data)
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = input_data.shape[0]
    for i in range(L-tw):
        train_seq = input_data[i:i+tw,]
        train_label = input_data[i+tw:i+tw+1,0]#预测time_step之后的第一个数值
        inout_seq.append((train_seq, train_label))#inout_seq内的数据不断更新，但是总量只有tw+1个
    return inout_seq
train_window = 12#设置训练输入的序列长度为12，类似于time_step = 12
train_inout_seq = create_inout_sequences(train_data, train_window)
print(train_inout_seq[:5])#产看数据集改造结果


# def load_cora_data():
#     data = citegrh.load_cora()
#     print(data.graph)
#
#     features = torch.FloatTensor(data.features)
#     labels = torch.LongTensor(data.labels)
#     mask = torch.BoolTensor(data.train_mask)
#     g = DGLGraph(data.graph)
#
#     return  g,features, labels, mask
#
# g, features, labels, mask =  load_cora_data()
#
#
#
# data = citegrh.load_cora()
# print(data.labels)

net = GAT_LSTM(g,1, hidden_dim=1, out_dim=1, num_heads=2)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
dur = []

# print(net)
for epoch in range(50):
    for seq,labels in train_inout_seq:
        # print('test1')
        optimizer.zero_grad()
        net.lstm.hidden_cell = (torch.zeros(1, 1, net.lstm.hidden_layer_size),
                             torch.zeros(1, 1, net.lstm.hidden_layer_size))
        # 实例化模型
        # print('test2')
        y_pred = net(seq)
        # print('test3')
        # 计算损失，反向传播梯度以及更新模型参数
        single_loss = loss_function(y_pred, labels)  # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
        single_loss.backward()  # 调用loss.backward()自动生成梯度，
        optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

    print(f'epoch:{epoch:3} loss:{single_loss.item():10.8f}')

torch.save(net,"GAT_LSTM.pkl")

pre_model=torch.load("GAT_LSTM.pkl")
pred_list=[]
for i in range(12):
    data_for_pred=all_data[:(-12+i),:]
    pred_list.append(data_for_pred[(-12):,:])

pre_model.eval()
pred=[]

for i in range(12):
    pred.append(float(pre_model(pred_list[i])[0]))

pre=np.array(pred)
pd.DataFrame(pre).to_csv("GAT_LSTM_pred.csv")
all_data[(-12):,0]
pre
np.mean(abs(pre-all_data[(-12):,0]))

# pred=[]
# for_pred=torch.zeros(size=(18,9))
# for i in range(12):
#     for_pred[0:12,:]=pred_list[i]
#     for j in range(6):
#         pred.append(float(pre_model(for_pred[(0+j):(12+j),:])[0]))
#         for_pred[12+i,:]=

'''
常见问题：
1、出现runtimeError：expected scalar type Double but found Float
具体出现语句：lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1).float(), self.hidden_cell)
如果写成：lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
则会出现以上异常，问题主要在于input_seq.view(len(input_seq), 1, -1)和self.hidden_cell的数据类型不一致，
前者为Float64，也就是Double，后者是Float32，通过tensorName.float()把float64转化为Float32后问题解决
2、tensor数据转化为numpy：tensorName.numpy()
3、numpy转化为tensor：torch.tensor(ndarray)
4、dataframe转化成为numpy：dataframeName.values
5、tensorName.view()的使用:
'''
