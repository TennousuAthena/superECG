import h5py
from scipy import io
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



def save_matlab_to_csv(matlab_file_path, csv_file_path):
    # 打开MATLAB文件
    data = h5py.File(matlab_file_path, 'r')

    # 获取数据集中的键（类似于MATLAB中的变量名）
    keys = list(data.keys())

    # 循环遍历键，将数据保存到CSV文件中
    for key in keys:
        dataset = data[key]

        # 如果数据集是一个二维数组
        if len(dataset.shape) == 2:
            # 转换为DataFrame
            df = pd.DataFrame(np.array(dataset))

            # 保存到CSV文件
            df.to_csv(csv_file_path, index=False)
            print(f"Data from {matlab_file_path} has been successfully saved to {csv_file_path}!")
# 保存 train_af.mat 数据到 df_af.csv
save_matlab_to_csv('/kaggle/input/bmedatas/Training/Training/train_af.mat', 'df_af.csv')

# 保存 train_nor.mat 数据到 df_nor.csv
save_matlab_to_csv('/kaggle/input/bmedatas/Training/Training/train_nor.mat', 'df_nor.csv')




df_nor = pd.read_csv('df_nor.csv')
df_af = pd.read_csv('df_af.csv')

df_norT = df_nor.transpose()
df_afT = df_af.transpose()

df_norT['label'] = 0  # 正常数据标签为0
df_afT['label'] = 1   # 房颤数据标签为1


print(df_norT.head())
print("-------------------------------------------------------------------")
print(df_afT.head())



df = pd.concat([df_norT, df_afT], ignore_index=True)


print(df.head())



print("-------------------------------------------------------------------")
# 查看DataFrame的基本信息，包括列名、数据类型和缺失值等
print("\nDataFrame的基本信息：")
print(df.info())



# 假设df是你的DataFrame变量名
data = df.iloc[0, :4000]  # 提取第一行的前4000个数字

plt.plot(data)
plt.title('Plot of First Row of DataFrame')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()



import os
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 加载数据函数
def loadData():
    # 这里假设你的数据已经加载到 DataFrame df 中，特征列为 X，标签列为 Y
    # 可以根据实际情况修改
    X = df.iloc[:, :-1].values  # 特征
    Y = df.iloc[:, -1].values   # 标签

    # 使用标签编码将标签转换为数字
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, Y_train, X_test, Y_test



# 构建CNN模型
def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4000, 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='same', activation='relu'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='same', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='same', activation='relu'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='same'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='same', activation='relu'),
        # 打平层,方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newModel


def loadData():
    # 这里假设你已经将数据加载到名为 df 的 DataFrame 中
    # X 是特征数据，Y 是标签数据
    X = df.iloc[:, :-1].values  # 假设最后一列是标签列
    Y = df.iloc[:, -1].values

    # 将数据分割成训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 将数据 reshape 成模型需要的输入形状 (300, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, Y_train, X_test, Y_test



def main():
    # 加载数据
    X_train, Y_train, X_test, Y_test = loadData()

    model_path = "your_model_path/model.h5"  # 模型保存路径
    log_dir = "your_log_dir"  # TensorBoard 日志保存路径
    RATIO = 0.2  # 验证集比例

    if os.path.exists(model_path):
        # 导入训练好的模型
        model = tf.keras.models.load_model(filepath=model_path)
    else:
        # 构建并编译模型
        model = buildModel()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # 定义TensorBoard对象
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # 训练与验证
        model.fit(X_train, Y_train, epochs=30,
                  batch_size=128,
                  validation_split=RATIO,
                  callbacks=[tensorboard_callback])
        model.save(filepath=model_path)

    # 预测
    Y_pred = model.predict_classes(X_test)

    
if __name__ == "__main__":
    main()