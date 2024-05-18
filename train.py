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
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False


# 1.开始对模型进行训练
input_size = 21  
hidden_size = 64  
batch_first = True  
epoch = 100  
totall_loss = []  
num_layers = 1  
batch_size = 32  
seq_length = 5  

# 2.定义获取数据函数，数据预处理。
from tqdm import tqdm  

def getData(root, sequence_length, batch_size):
    stock_data = pd.read_excel(root)  
    print(stock_data.info())  
    print(stock_data.head().to_string())  
    # 2.1对数据进行标准化min-max
    scaler = MinMaxScaler()  
    df = scaler.fit_transform(stock_data) 
    print("整理后\n", df) 

    # 2.2构造X,Y
    sequence = sequence_length  
    x = []  
    y = []  
    for i in range(df.shape[0] - sequence):
        x.append(df[i:i + sequence, :])  
        y.append(df[i + sequence, :])  
    x = np.array(x, dtype=np.float32)  
    y = np.array(y, dtype=np.float32)

    print("x.shape=", x.shape)  
    print("y.shape", y.shape) 

    # 2.3构造batch,构造训练集train与测试集test
    total_len = len(y)  
    print("total_len=", total_len)
    trainx, trainy = x[:int(0.80 * total_len), ], y[:int(0.80 * total_len), ]  
    testx, testy = x[int(0.80 * total_len):, ], y[int(0.80 * total_len):, ]  

    train_loader = DataLoader(dataset=Mydataset(trainx, trainy), shuffle=True, batch_size=batch_size)  
    test_loader = DataLoader(dataset=Mydataset(testx, testy), shuffle=False, batch_size=batch_size)  
    return [scaler, train_loader, test_loader]  



# 3.自己重写数据集继承Dataset
class Mydataset(Dataset):  
    def __init__(self, x, y):  
        self.x = torch.from_numpy(x) 
        self.y = torch.from_numpy(y)  

    def __getitem__(self, index):  
        x1 = self.x[index]  
        y1 = self.y[index]  
        return x1, y1  

    def __len__(self):  
        return len(self.x)  

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  


root = r".\关节点00.xlsx"  
scaler, train_loader, test_loader = getData(root, sequence_length=24, batch_size=64)  
print("train_loader=", len(train_loader)) 
print("test_loader=", len(test_loader))  



# 4.构建LSTM模型
import torch.nn as nn

class lstm(nn.Module): 
    def __init__(self, input_size=8, hidden_size=32, num_layer=1, dropout=0.2, batch_first=True):
        super(lstm, self).__init__()  
        self.hidden_size = hidden_size  
        self.input_size = input_size  
        self.num_layer = num_layer  
        self.dropout = dropout 
        self.batch_first = batch_first  
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layer, batch_first=batch_first, dropout=self.dropout)
        self.linear1 = nn.Linear(self.hidden_size, 2 * self.hidden_size)  
        self.linear2 = nn.Linear(self.hidden_size * 2, self.input_size)  

    def forward(self, x):  # 定义模型的前向传播
        out, (hidden, cell) = self.lstm(x)  
        out = self.linear1(out)  
        out = self.linear2(out)  
        out = out[:, -1, :]  
        return out


def train():  # 定义训练函数
    model = lstm(input_size=input_size, hidden_size=hidden_size, num_layer=num_layers) 
    model.to(device)  
    criterion = nn.MSELoss(reduction="mean")  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    epoch_losses = []  

    for i in range(epoch):  
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)  #
        total_loss = 0  
        for idx, data in loop:
            x, y = data  
            x = x.to(device)  
            y = y.to(device)  
            pred = model(x)  
            loss = criterion(pred, y) 
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            total_loss += loss.item() * x.size(0)  
            loop.set_description(f'Epoch: [{i + 1}/{epoch}]')  
            loop.set_postfix(loss=loss.item())  

        epoch_loss = total_loss / len(train_loader.dataset)  
        epoch_losses.append(epoch_loss)  


    plt.figure(figsize=(10, 5))  
    plt.plot(epoch_losses, label='Training Loss') 
    plt.title('Training Loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.legend()  
    plt.show()  

    return model  

model = train()  

torch.save(model, "./weight.pt")  




