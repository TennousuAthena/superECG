import torch
import torch.nn as nn

class Conv1dModel(nn.Module):
    def __init__(self, class_num=4, fc_len=72000):
        super(Conv1dModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=21, stride=1, padding=10)  # padding='same' in keras equals padding=(kernel_size-1)//2 in pytorch
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=23, stride=1, padding=11)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=25, stride=1, padding=12)
        self.avgpool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=27, stride=1, padding=13)
        self.fc1 = nn.Linear(fc_len, 128) #32000
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, class_num)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.relu(self.conv3(x))
        x = self.avgpool1(x)
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.softmax(self.fc2(x), dim=1)
        return x