import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPEncoder, self).__init__()
        # 定义一个线性层，将输入数据映射到隐藏层
        self.linear1 = nn.Linear(input_size, hidden_size)
        # 定义一个激活函数，这里使用ReLU函数
        self.relu = nn.ReLU()
        # 定义一个线性层，将隐藏层映射到输出层，即编码向量
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 前向传播，计算编码向量
        x = self.linear1(x) # 计算隐藏层
        x = self.relu(x) # 应用激活函数
        x = self.linear2(x) # 计算输出层，即编码向量
        return x

class MLPDecoder(nn.Module):
    def __init__(self, input_size,out_size, hidden_size) -> None:
        super(MLPDecoder,self).__init__()