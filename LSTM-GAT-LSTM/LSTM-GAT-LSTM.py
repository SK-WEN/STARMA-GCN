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
    def __init__(self, g,feature_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.attn_fc = nn.Linear(2*feature_dim, 1, bias=False)
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
        self.g.ndata['z'] = h # 每个节点的特征
        self.g.apply_edges(self.edge_attention) # 为每一条边获得其注意力系数
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, feature_dim , num_heads=1, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, feature_dim))
        self.merge = merge


    def forward(self, h):
        head_out = [attn_head(h) for attn_head in self.heads]
        if self.merge=='cat':
            return torch.cat(head_out, dim=1)
        else:
            return torch.mean(torch.stack(head_out))

class LSTM_GAT(nn.Module):
    def __init__(self,LSTM_layers,input_size,hidden_layer_size,g, initialize=0,num_heads=1, merge='avg'):
        super(LSTM_GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.ini=initialize
        self.hidden_layer_size=hidden_layer_size
        self.h=0
        self.c=0
        self.hidden_cell=(
            torch.zeros(1,1,hidden_layer_size),
            torch.zeros(1,1,hidden_layer_size)
        )
        for i in range(LSTM_layers):
            self.layers.append(nn.LSTM(input_size,hidden_layer_size))
        self.GAT=MultiHeadGATLayer(g, hidden_layer_size , num_heads, merge)
        self.hidden_cell=(
            torch.zeros(1,1,hidden_layer_size),
            torch.zeros(1,1,hidden_layer_size)
        )

    def forward(self,data):
        h=torch.zeros(data.shape[0],self.hidden_layer_size)
        c=torch.zeros(data.shape[0],self.hidden_layer_size)
        if self.ini==0:
            for i in range(data.shape[0]):
                _,output=self.layers[i](data[i,].view(-1,1,1),self.hidden_cell)
                c[i]=output[1]
                h[i]=output[0]
        else:
            for i in range(data.shape[0]):
                _, output = self.layers[i](data[i,].view(-1,1,1), (self.h[i,],self.c[i,]))
                c[i] = output[1]
                h[i] = output[0]
        h=self.GAT(h)
        return h,c

class multi_LSTM_GAT(nn.Module):
    def __init__(self,time_step,LSTM_layers,input_size,hidden_layer_size,g, output_size=1,num_heads=1, merge='avg'):
        super(multi_LSTM_GAT, self).__init__()
        self.step=nn.ModuleList()
        for i in range(time_step):
            self.step.append(LSTM_GAT(LSTM_layers,input_size,hidden_layer_size,g, initialize=i,num_heads=num_heads, merge=merge))
        self.fc=nn.Linear(hidden_layer_size,output_size)

    def forward(self,data):
        count = 0
        while True:
            h,c=self.step[count](data[:,:,count])
            count = count + 1
            if count>=len(self.step): break
            self.step[count].h=h
            self.step[count].c=c
        return(self.fc(h))

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

#定义create_inout_sequences函数，接收原始输入数据，并返回一个元组列表。
train_data=torch.FloatTensor(train_data)
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = input_data.shape[0]
    for i in range(L-tw):
        train_seq = input_data[i:i+tw,]
        train_label = input_data[i+tw:i+tw+1,]#预测time_step之后的第一个数值
        inout_seq.append((train_seq, train_label))#inout_seq内的数据不断更新，但是总量只有tw+1个
    return inout_seq
train_window = 3#设置训练输入的序列长度为12，类似于time_step = 12
train_inout_seq = create_inout_sequences(train_data, train_window)
print(train_inout_seq[:5])#产看数据集改造结果

net = LSTM_GAT(time_step=3,LSTM_layers=9,input_size=1,hidden_layer_size=9,g=g, output_size=1,num_heads=1, merge='avg')
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
dur = []

# print(net)
for epoch in range(50):
    for seq,labels in train_inout_seq:
        # print('test1')
        optimizer.zero_grad()
        # 实例化模型
        # print('test2')
        y_pred = net(seq.view(seq.shape[1],-1,seq.shape[0]))
        # print('test3')
        # 计算损失，反向传播梯度以及更新模型参数
        single_loss = loss_function(y_pred, labels)  # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
        single_loss.backward()  # 调用loss.backward()自动生成梯度，
        optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

    print(f'epoch:{epoch:3} loss:{single_loss.item():10.8f}')

torch.save(net,"LSTM_GAT.pkl")

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
