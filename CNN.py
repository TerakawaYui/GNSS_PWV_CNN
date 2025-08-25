# CNN.py - PyTorch Version
import torch
import torch.nn as nn
import torch.nn.functional as F

class PWVCNN(nn.Module):
    """
    根据论文描述构建PyTorch CNN模型。
    """
    def __init__(self, input_features):
        """
        Args:
            input_features (int): 输入数据的特征数量 (即sequence_length)。
                                  例如，如果输入形状是 (batch_size, 1, 10)，则 input_features=10。
        """
        super(PWVCNN, self).__init__()

        # PyTorch Conv1d 期望输入形状 (batch_size, in_channels, sequence_length)
        # 我们的数据是 (batch_size, 1, input_features)，所以 in_channels=1

        # 第一个卷积层
        # in_channels=1 (因为我们将每个样本视为1个通道，长度为特征数)
        # out_channels=30 (对应论文中的30个核)
        # kernel_size=4 (对应论文中的4x1卷积核的第一个维度4)
        # stride=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=30, kernel_size=4, stride=1, padding=0)
        # 激活函数ReLU
        self.relu1 = nn.ReLU()
        # 第一个池化层
        # kernel_size=2 (对应论文中的池化核大小为2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 第二个卷积层
        # in_channels=30 (来自前一个卷积层的输出通道数)
        # out_channels=30 (沿用30个核)
        # kernel_size=2 (论文中未明确，这里假设为2)
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=2, stride=1, padding=0)
        # 激活函数ReLU
        self.relu2 = nn.ReLU()
        # 第二个池化层
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 计算全连接层输入维度
        # 假设输入特征是 input_features (例如10)
        # conv1输出长度: (input_features - kernel_size + 2*padding) / stride + 1
        # (input_features - 4 + 0) / 1 + 1 = input_features - 3
        # pool1输出长度: (input_features - 3) / 2 (向下取整)
        # conv2输出长度: (pool1_output_length - 2 + 0) / 1 + 1 = pool1_output_length - 1
        # pool2输出长度: (conv2_output_length) / 2 (向下取整)

        # 为了动态计算展平后的维度，我们需要一个假输入来通过网络计算一次
        # 假设输入是 (1, 1, input_features)
        dummy_input = torch.randn(1, 1, input_features)
        x = self.pool1(self.relu1(self.conv1(dummy_input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        self.flattened_size = x.view(x.size(0), -1).size(1)

        # 全连接层
        # 论文图示中，全连接层输入30个节点，输出1个
        self.fc1 = nn.Linear(self.flattened_size, 30) # 对应论文图示中的30个节点
        self.relu3 = nn.ReLU()
        self.fc_output = nn.Linear(30, 1) # 回归任务，输出1个值

    def forward(self, x):
        # x 形状: (batch_size, 1, input_features)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)

        # 展平
        x = x.view(x.size(0), -1) # 将 (batch_size, channels, length) 展平为 (batch_size, channels * length)

        x = self.relu3(self.fc1(x))
        x = self.fc_output(x)
        return x

if __name__ == '__main__':
    # 示例用法：假设输入特征长度为10
    input_features = 10
    model = PWVCNN(input_features)
    print(model)

    # 假输入测试
    dummy_input = torch.randn(1, 1, input_features) # (batch_size, channels, sequence_length)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # 应该输出 (1, 1)