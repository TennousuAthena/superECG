# %% 数据预处理
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchkeras, torchmetrics
from utils import pltImage
import matplotlib
# Only Apple can do it
if os.name == 'posix':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dataset import BMEData, CardiacArrhythmia

# data = BMEData(batchSize=256)
data = CardiacArrhythmia(batchSize=256,)
dl_train, dl_val = data.getDataLoader()
X_batch, Y_batch = data.getSingleBatch(dl_train)
print(X_batch.shape, Y_batch.shape)
singleLine = X_batch[1].squeeze().numpy()
pltImage(pd.DataFrame(singleLine))

#%% 模型训练
class Conv1dModel(nn.Module):
    def __init__(self):
        super(Conv1dModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=21, stride=1, padding=10)  # padding='same' in keras equals padding=(kernel_size-1)//2 in pytorch
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=23, stride=1, padding=11)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=25, stride=1, padding=12)
        self.avgpool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(32, 64, kernel_size=27, stride=1, padding=13)
        self.fc1 = nn.Linear(72000, 128) #32000
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 4) # 2
        
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

net = Conv1dModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

model = torchkeras.KerasModel(net,
                                loss_fn = loss_fn,
                                optimizer= optimizer,
                                metrics_dict = {"acc":torchmetrics.Accuracy(task='binary'),},
                                 )
torchkeras.summary(model,input_shape=X_batch[0].shape)

dfhistory=model.fit(train_data=dl_train, 
                    val_data=dl_val, 
                    epochs=100, 
                    patience=10, 
                    ckpt_path='checkpoint',
                    monitor="val_acc",
                    mode="max",
                    plot=True
                   )
