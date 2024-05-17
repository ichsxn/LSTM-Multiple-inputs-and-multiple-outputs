# 1.导入库 对数据集进行处理
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'#绘图正常显示中文
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号#有中文出现的情况，


# 1.开始对模型进行训练
input_size = 21  # 输入特征列的数量
hidden_size = 64  # 隐藏层的大小
batch_first = True  # 是否在数据的表示中将batch_size放在第一位
epoch = 100  # 训练的总轮次

totall_loss = []  # 用于记录每一轮的损失值
num_layers = 1  # RNN的层数
batch_size = 32  # 每个批次的样本数
seq_length = 5  # 序列的长度

# 2.定义获取数据函数，数据预处理。
from tqdm import tqdm  # 导入进度条库

def getData(root, sequence_length, batch_size):
    stock_data = pd.read_excel(root)  # 从excel文件中读取股票数据
    print(stock_data.info())  # 打印数据的基本信息
    print(stock_data.head().to_string())  # 打印前五行数据

    # 2.1对数据进行标准化min-max
    scaler = MinMaxScaler()  # 创建MinMaxScaler对象进行标准化
    df = scaler.fit_transform(stock_data)  # 对数据进行标准化处理
    print("整理后\n", df)  # 打印标准化后的数据

    # 2.2构造X,Y
    sequence = sequence_length  # 序列长度
    x = []  # 存储特征数据
    y = []  # 存储标签数据
    for i in range(df.shape[0] - sequence):
        x.append(df[i:i + sequence, :])  # 取出序列作为特征
        y.append(df[i + sequence, :])  # 取出序列后一项作为标签
    x = np.array(x, dtype=np.float32)  # 将列表转换为numpy数组，指定数据类型为float32
    y = np.array(y, dtype=np.float32)  # 同上

    print("x.shape=", x.shape)  # 打印特征数据的形状
    print("y.shape", y.shape)  # 打印标签数据的形状

    # 2.3构造batch,构造训练集train与测试集test
    total_len = len(y)  # 总数据长度
    print("total_len=", total_len)
    trainx, trainy = x[:int(0.80 * total_len), ], y[:int(0.80 * total_len), ]  # 划分80%的数据作为训练集
    testx, testy = x[int(0.80 * total_len):, ], y[int(0.80 * total_len):, ]  # 剩余的20%作为测试集

    train_loader = DataLoader(dataset=Mydataset(trainx, trainy), shuffle=True, batch_size=batch_size)  # 创建训练数据的DataLoader
    test_loader = DataLoader(dataset=Mydataset(testx, testy), shuffle=False, batch_size=batch_size)  # 创建测试数据的DataLoader
    return [scaler, train_loader, test_loader]  # 返回标准化工具、训练数据加载器和测试数据加载器



# 3.自己重写数据集继承Dataset
class Mydataset(Dataset):  # 创建一个数据集类，继承自Dataset
    def __init__(self, x, y):  # 初始化函数，接收特征数据x和标签数据y
        self.x = torch.from_numpy(x)  # 将numpy数组x转换为torch张量
        self.y = torch.from_numpy(y)  # 将numpy数组y转换为torch张量

    def __getitem__(self, index):  # 定义获取单个样本的方法
        x1 = self.x[index]  # 根据索引获取特征数据
        y1 = self.y[index]  # 根据索引获取标签数据
        return x1, y1  # 返回一对特征和标签

    def __len__(self):  # 定义获取数据集大小的方法
        return len(self.x)  # 返回特征数据的长度，即样本数量

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 确定使用的设备是GPU还是CPU
#device = torch.device('cpu')  # 直接指定使用CPU

root = r".\关节点00.xlsx"  # 设置数据文件的路径
scaler, train_loader, test_loader = getData(root, sequence_length=24, batch_size=64)  # 调用getData函数获取数据加载器
print("train_loader=", len(train_loader))  # 打印训练数据加载器中批次的数量
print("test_loader=", len(test_loader))  # 打印测试数据加载器中批次的数量



# 4.构建LSTM模型
import torch.nn as nn

class lstm(nn.Module):  # 定义一个LSTM模型类
    def __init__(self, input_size=8, hidden_size=32, num_layer=1, dropout=0.2, batch_first=True):
        super(lstm, self).__init__()  # 调用父类的初始化函数
        self.hidden_size = hidden_size  # 隐藏层的大小
        self.input_size = input_size  # 输入层的大小
        self.num_layer = num_layer  # LSTM层的数量
        self.dropout = dropout  # dropout比例
        self.batch_first = batch_first  # 数据的第一个维度是否为batch size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layer, batch_first=batch_first, dropout=self.dropout)
        self.linear1 = nn.Linear(self.hidden_size, 2 * self.hidden_size)  # 定义第一个全连接层
        self.linear2 = nn.Linear(self.hidden_size * 2, self.input_size)  # 定义第二个全连接层

    def forward(self, x):  # 定义模型的前向传播
        out, (hidden, cell) = self.lstm(x)  # LSTM层的输出
        out = self.linear1(out)  # 第一个全连接层的输出
        out = self.linear2(out)  # 第二个全连接层的输出
        out = out[:, -1, :]  # 只取序列的最后一个时间点的输出
        return out


def train():  # 定义训练函数
    model = lstm(input_size=input_size, hidden_size=hidden_size, num_layer=num_layers)  # 实例化LSTM模型
    model.to(device)  # 将模型转移到指定的设备上
    criterion = nn.MSELoss(reduction="mean")  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
    epoch_losses = []  # 定义一个列表来存储每个epoch的平均损失

    for i in range(epoch):  # 训练循环
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)  # 创建一个进度条
        total_loss = 0  # 初始化总损失为0
        for idx, data in loop:
            x, y = data  # 获取数据和标签
            x = x.to(device)  # 将数据转移到指定的设备上
            y = y.to(device)  # 将标签转移到指定的设备上

            pred = model(x)  # 模型预测
            loss = criterion(pred, y)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item() * x.size(0)  # 累计损失

            loop.set_description(f'Epoch: [{i + 1}/{epoch}]')  # 更新进度条的描述
            loop.set_postfix(loss=loss.item())  # 在进度条后显示当前损失

        epoch_loss = total_loss / len(train_loader.dataset)  # 计算这个epoch的平均损失
        epoch_losses.append(epoch_loss)  # 存储这个epoch的平均损失


    plt.figure(figsize=(10, 5))  # 设置图的大小
    plt.plot(epoch_losses, label='Training Loss')  # 绘制损失曲线
    plt.title('Training Loss')  # 设置图的标题
    plt.xlabel('Epochs')  # X轴标签
    plt.ylabel('Loss')  # Y轴标签
    plt.legend()  # 显示图例
    plt.show()  # 显示图

    return model  # 返回训练好的模型

model = train()  # 训练模型

torch.save(model, "./weight.pt")  # 保存模型权重
