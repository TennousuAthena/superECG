import os
import h5py
import torch
import scipy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Tuple
import matplotlib
if os.name == 'posix':
    matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class BaseDataset:
    def __init__(self, basePath: str = "./Training", batchSize: int = 32, test_size: float = 0.2) -> None:
        super().__init__()
        self.basePath = os.path.abspath(basePath) + "/"
        self.batchSize = batchSize
        self.test_size = test_size
        
    def cacheData(self, data: pd.DataFrame, filename: str) -> None:
        if not os.path.exists("cache"):
            os.makedirs("cache")
        if not os.path.exists("cache/" + filename):
            os.makedirs("cache/" + filename)
        with open("cache/" +filename+".p", 'wb') as f:
            pickle.dump(data, f)
            
    def checkCache(self, filename: str) -> bool:
        return os.path.exists("cache/" +filename+".p")
    
    def getCache(self, filename: str) -> pd.DataFrame:
        with open("cache/" +filename+".p", 'rb') as f:
            return pickle.load(f)
        
    def getSingleBatch(self, dl_train: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        for X_batch, Y_batch in dl_train:
            return X_batch, Y_batch

class BMEData(BaseDataset):
    def __init__(self, basePath: str = "./bmedesign-ecg/Training", batchSize: int = 32, test_size: float = 0.2) -> None:
        super().__init__(basePath, batchSize, test_size)

    def readMatAsDF(self, file_path: str) -> pd.DataFrame:
        data = h5py.File(file_path, 'r')
        keys = list(data.keys())
        assert len(keys) == 1
        for key in keys:
            dataset = data[key]
            if len(dataset.shape) == 2:
                df = pd.DataFrame(np.array(dataset))
                return df

    def getDataLoader(self) -> Tuple[DataLoader, DataLoader]:
        X_normal = self.readMatAsDF(self.basePath + "train_nor.mat")
        X_af = self.readMatAsDF(self.basePath + "train_af.mat")
        X_normal, X_af = X_normal.T, X_af.T
        Y_normal = np.zeros((X_normal.shape[0], 1))
        Y_af = np.ones((X_af.shape[0], 1))

        X = np.concatenate((X_normal, X_af), axis=0)
        Y = np.concatenate((Y_normal, Y_af), axis=0)
        
        Y = pd.get_dummies(Y[:, 0]).values.astype(np.float32)

        del X_normal, X_af, Y_normal, Y_af
        
        print(X.shape, Y.shape)

        # 数据集划分
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=self.test_size, random_state=42)
        X_train = torch.tensor(X_train).float().unsqueeze(1)
        X_val = torch.tensor(X_val).float().unsqueeze(1)
        Y_train = torch.tensor(Y_train).float()
        Y_val = torch.tensor(Y_val).float()

        ds_train = TensorDataset(X_train, Y_train)
        ds_val = TensorDataset(X_val, Y_val)

        dl_train = DataLoader(ds_train, batch_size=self.batchSize, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=self.batchSize, shuffle=True)

        print(X_train.shape, Y_train.shape)

        del X, Y, ds_train, ds_val

        return dl_train, dl_val


class CardiacArrhythmia(BaseDataset):
    def __init__(self, basePath: str = "./CardiacArrhythmia", batchSize: int = 32, test_size: float = 0.2, sample_size = 9000) -> None:
        self.sample_size = sample_size
        self.labels = ["N", "A", "O", "~"]
        self.lb =  LabelBinarizer()
        self.lb.fit(self.labels)
        super().__init__(basePath, batchSize, test_size)
    
    def readMatAsNP(self, file_path: str) -> pd.DataFrame:
        data = scipy.io.loadmat(file_path)
        assert len(data['val']) == 1
        return data['val']
    
    """
    当前策略:读取4个参考文件,将所有的数据合并,然后按照文件名进行分组,取首个众数作为标签
    """
    def loadTrainAnnotation(self) -> pd.DataFrame:
        annotationFiles = [f"REFERENCE-v{i}.csv" for i in range(4)]
        Annotations = []
        for anfile in annotationFiles:
            data = pd.read_csv(self.basePath + anfile, header=None)
            Annotations.append(data)
        all_data = pd.concat(Annotations)
        grouped = all_data.groupby(0)[1].agg(lambda x: pd.Series.mode(x)[0])
        result = pd.DataFrame(grouped).reset_index()
        return result
    
    def y2tensor(self, y: str) -> torch.Tensor:
        return torch.tensor(self.lb.transform([y])[0]).float()
    
    def tensor2y(self, one_hot: torch.Tensor) -> str:
        return self.lb.inverse_transform(np.array([one_hot]))
    
    def formatData(self, filePath: str, fileList: list, annotation : pd.DataFrame) -> pd.DataFrame:
        if self.checkCache(filePath):
            return self.getCache(filePath)
        Data = pd.DataFrame(columns=['name', 'data', 'label'])
        for file in fileList:
            if file.endswith(".mat"):
                data = self.readMatAsNP(self.basePath + filePath + file)[0]
                if data.size != self.sample_size:      # Resample data
                    data = resample(data, self.sample_size)
                new_row = pd.Series({'name': file, 'data': data, 'label': annotation[annotation[0] == file[:-4]][1].values[0]})
                Data.loc[Data.shape[0]] = new_row
        self.cacheData(Data, filePath)
        return Data
    
    def getDataLoader(self) -> Tuple[DataLoader, DataLoader]:
        train_path = "training2017/"
        val_path = "sample2017/validation/"
        train_files = os.listdir(self.basePath + train_path)
        val_files = os.listdir(self.basePath + val_path)
        TrainAnnotation = self.loadTrainAnnotation()
        ValAnnotation = pd.read_csv(self.basePath + "sample2017/answers.txt", header=None)
        TrainData = self.formatData(train_path, train_files, TrainAnnotation)
        ValData = self.formatData(val_path, val_files, ValAnnotation)
        
        xtrain = TrainData['data'].values.tolist()
        xtrain = np.array(xtrain)
        xtrain = torch.tensor(xtrain).float().unsqueeze(1)
        ytrain_list = [tensor.numpy().tolist() for tensor in TrainData['label'].apply(self.y2tensor)]  # Convert series of tensors to list of lists
        ytrain = torch.tensor(ytrain_list)  # Convert list of lists to 2D tensor
        
        xval = ValData['data'].values.tolist()
        xval = np.array(xval)
        xval = torch.tensor(xval).float().unsqueeze(1)
        yval_list = [tensor.numpy().tolist() for tensor in ValData['label'].apply(self.y2tensor)]
        yval = torch.tensor(yval_list)

        print(xtrain.shape, ytrain.shape)
        train_ds = TensorDataset(xtrain, ytrain)
        val_ds = TensorDataset(xval, yval)
        
        train_dl = DataLoader(train_ds, batch_size=self.batchSize, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batchSize, shuffle=True)
        return train_dl, val_dl
    
if __name__ == "__main__":
    testData = CardiacArrhythmia()
    testData.getDataLoader()