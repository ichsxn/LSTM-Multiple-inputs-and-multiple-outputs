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


#1.开始对模型进行训练
input_size = 21  # 输入列的数量
hidden_size = 64  # 隐藏层的大小
output_size = 1  # 输出层的大小
batch_first = True  # 如果True，则输入和输出张量的形状为 (batch, seq, feature)
epoch = 1  # 训练轮数

totall_loss = []  # 用于记录每个epoch的损失值
num_layers=1  # 网络层数
batch_size=32  # 批处理大小
seq_length=5  # 序列的长度



# 2.定义获取数据函数，数据预处理。

def getData(root, sequence_length, batch_size):
    stock_data = pd.read_excel(root)  # 从Excel文件读取股票数据
    print(stock_data.info())  # 打印数据信息，包括列名和数据类型等
    print(stock_data.head().to_string())  # 打印前5行数据，以字符串格式

    # 2.1对数据进行标准化处理 (min-max scaling)
    scaler = MinMaxScaler()  # 创建一个MinMaxScaler对象
    df = scaler.fit_transform(stock_data)  # 将股票数据标准化
    print("整理后\n", df)  # 打印标准化后的数据

    # 2.2构造输入X和标签Y
    sequence = sequence_length
    x = []  # 存储输入序列
    y = []  # 存储输出序列（标签）
    for i in range(df.shape[0] - sequence):
        x.append(df[i:i + sequence, :])  # 生成输入序列
        y.append(df[i + sequence, :])  # 生成对应的标签
    x = np.array(x, dtype=np.float32)  # 转换为numpy数组
    y = np.array(y, dtype=np.float32)  # 转换为numpy数组

    print("x.shape=", x.shape)  # 打印x的形状
    print("y.shape", y.shape)  # 打印y的形状

    # 2.3划分训练集和测试集，构造批处理
    total_len = len(y)  # 总的数据长度
    print("total_len=", total_len)  # 打印总长度
    trainx, trainy = x[:int(0.80 * total_len), ], y[:int(0.80 * total_len), ]  # 划分训练数据
    testx, testy = x[int(0.80 * total_len):, ], y[int(0.80 * total_len):, ]  # 划分测试数据

    # 使用自定义的数据集类（Mydataset）和DataLoader来加载数据
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy), shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), shuffle=False, batch_size=batch_size)
    return [scaler, train_loader, test_loader]  # 返回标准化工具、训练数据加载器和测试数据加载器



# 3.自己重写数据集继承Dataset
class Mydataset(Dataset):  # 继承自PyTorch的Dataset类
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)  # 将numpy数组转换为torch张量
        self.y = torch.from_numpy(y)  # 将numpy数组转换为torch张量

    def __getitem__(self, index):  # 通过索引获取数据集中的一项
        x1 = self.x[index]  # 获取输入张量
        y1 = self.y[index]  # 获取标签张量
        return x1, y1  # 返回一对输入和标签

    def __len__(self):  # 获取数据集中元素的总数
        return len(self.x)  # 返回x的长度


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 确定使用的设备是GPU还是CPU
#device = torch.device('cpu')  # 直接指定使用CPU

root = r".\关节点00.xlsx"  # 定义数据文件的路径
scaler, train_loader, test_loader = getData(root, sequence_length=24, batch_size=64)  # 获取数据加载器

print("train_loader=",len(train_loader))  # 打印训练数据加载器中批次的数量
print("test_loader=",len(test_loader))  # 打印测试数据加载器中批次的数量



#5.构建LSTM模型
import torch.nn as nn

class lstm(nn.Module):  # 继承自nn.Module
    def __init__(self, input_size=8, hidden_size=32, num_layer=1, dropout=0.2, batch_first=True):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size  # 隐藏层的大小
        self.input_size = input_size  # 输入层的大小
        self.num_layer = num_layer  # LSTM的层数
        self.output_size = output_size  # 输出层的大小
        self.dropout = dropout  # 在LSTM层之后应用的dropout比率
        self.batch_first = batch_first  # 输入输出张量的形状为(batch, seq, feature)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layer, batch_first=batch_first, dropout=self.dropout)  # 定义LSTM层
        self.linear1 = nn.Linear(self.hidden_size, 2*self.hidden_size)  # 定义第一个全连接层
        self.linear2 = nn.Linear(self.hidden_size*2, self.input_size)  # 定义第二个全连接层

    def forward(self, x):  # 定义前向传播
        out, (hidden, cell) = self.lstm(x)  # 通过LSTM层
        out = self.linear1(out)  # 通过第一个全连接层
        out = self.linear2(out)  # 通过第二个全连接层
        out = out[:,-1,:]  # 只获取序列的最后一个输出
        return out  # 返回输出



#6.MAPE和SMAPE
import numpy as np

def mape(y_true, y_pred):
    # 将输入转换为NumPy数组
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 检查真实值中是否存在零值，若存在则剔除，因为零值会导致除以零的错误
    mask = (y_true != 0)
    y_true, y_pred = y_true[mask], y_pred[mask]

    # 若过滤后的真实值为空，则无法计算MAPE，返回None
    if len(y_true) == 0:
        return None

    # 计算MAPE，添加一个极小值epsilon以避免除以零的错误
    epsilon = 1e-8
    absolute_percentage_error = np.abs((y_pred - y_true) / (y_true + epsilon))
    mape_value = np.mean(absolute_percentage_error) * 100  # 结果乘以100，转换为百分比形式

    return mape_value




# 7.构建测试验证函数

def evaluate(model):
    preds = []  # 存储所有预测值
    reals = []  # 存储所有真实值
    for idx, data in enumerate(test_loader):  # 遍历测试数据加载器
        x, y = data
        x = x.to(device)  # 将数据转移到指定的设备上（GPU或CPU）
        y = y.to(device)
        pred = model(x)  # 使用模型进行预测
        preds.append(pred.cpu().detach().numpy())  # 将预测结果转换为numpy数组，并存储
        reals.append(y.cpu().detach().numpy())  # 将真实标签转换为numpy数组，并存储
    preds = np.concatenate(preds, axis=0)  # 将预测结果列表合并为一个数组
    reals = np.concatenate(reals, axis=0)  # 将真实标签列表合并为一个数组
    preds=scaler.inverse_transform(preds)
    reals = scaler.inverse_transform(reals)

    # 打印形状信息以便调试
    print("preds=", preds.shape)
    print("reals=", reals.shape)


    # 对每个特征绘制预测值和真实值的对比图
    for i in range(preds.shape[1]):
        plt.figure(figsize=(8, 5))
        plt.title("特征——"+str(i)+"图")
        plt.plot(preds[:, i], label="预测值")
        plt.plot(reals[:, i], label="真实值")
        plt.legend()
        plt.savefig("./out/拟合图_"+str(i)+"_.png")  # 保存图像
        plt.show()

    # 计算并打印性能指标
    MAPE = mape(reals, preds)  # 计算MAPE
    r2 = r2_score(y_true=reals, y_pred=preds)  # 计算R2分数
    print("MAE=", mean_absolute_error(reals, preds))
    print("RMSE=", np.sqrt(mean_squared_error(reals, preds)))
    print("MAPE=", MAPE)
    print("R2=", r2)

model = torch.load("./weight.pt",map_location='cpu')  # 加载训练好的模型
evaluate(model)  # 使用加载的模型进行评估
