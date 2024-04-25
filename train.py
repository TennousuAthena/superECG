# %% 数据预处理
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from utils import pltImage
import torchkeras, torchmetrics
import matplotlib
# Only Apple can do it
if os.name == 'posix':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from model.conv1d import Conv1dModel
from dataset import BMEData, CardiacArrhythmia

# data = BMEData(batchSize=256)
data = CardiacArrhythmia(batchSize=256,)
dl_train, dl_val = data.getDataLoader()
X_batch, Y_batch = data.getSingleBatch(dl_train)
print(X_batch.shape, Y_batch.shape)
singleLine = X_batch[1].squeeze().numpy()
pltImage(pd.DataFrame(singleLine))

#%% 模型训练
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
