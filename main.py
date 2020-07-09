
"""
Anthor:liu jia ming
Date:2020/7
Theme:AQI Prediction
"""
#-*- coding: utf-8 -*-
from keras.optimizers import Adam
from keras.models import load_model
import  pandas as pd
import  numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from models.Model_CNN_LSTM_ATTENTION import attention_model
import matplotlib.pyplot as plt
from pyecharts import Bar
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset),look_back,-1):
        a = dataset[(i-look_back):i,:]
        dataX.append(a[::-1])
        dataY.append(dataset[i-look_back-1,:])
    X = np.array(dataX)
    y = np.array(dataY)
    return X, y



# 1.数据预处理
DAY_STEPS = 20
INPUT_DIMS = 7
lstm_units = 64
data = pd.read_csv('./data/data.csv',
                   index_col=0, encoding='gb2312')
scaler = preprocessing.MinMaxScaler()
index = data.index
col = data.columns
class_names = np.unique(data.iloc[:, -1])
X = scaler.fit_transform(data.iloc[:,])
y = scaler.fit_transform(data.iloc[:, -1].values.reshape(-1, 1))
X, _ = create_dataset(X,DAY_STEPS)
_ , y = create_dataset(y,DAY_STEPS)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=2020)

# 2.训练模型
m=load_model("model_600.h5")
# m = attention_model(DAY_STEPS = 20,INPUT_DIMS = 7,lstm_units = 64)
# m.summary()
# m.compile(optimizer=Adam(lr=0.00001), loss='mse')
# m.fit([X_train], y_train, epochs=200, batch_size=64, validation_split=0.1)

y_train_pred = scaler.inverse_transform(m.predict([X_train]))
y_test_pred = scaler.inverse_transform(m.predict([X_test]))
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

# 3.验证评估
bar = Bar()
trainList = []
trainList.append(r2_score(y_train,y_train_pred))
trainList.append(mean_squared_error(y_train, y_train_pred))
trainList.append(mean_absolute_error(y_train, y_train_pred))
trainList.append(explained_variance_score(y_train, y_train_pred))
trainList.append(1 - np.mean(np.abs(y_train_pred - y_train) / y_train))
bar.add('CNN-LSTM-ATTENTION训练集评估指标', ['r^2', '均方差', '绝对差', '解释度', '平均绝对比准确率'],
        np.round(trainList, 2), is_label_show=True, label_text_color='#000')
bar.render('CNN-LSTM-ATTENTION训练集评估指标.html')

valList = []
valList.append(r2_score(y_test,y_test_pred))
valList.append(mean_squared_error(y_test, y_test_pred))
valList.append(mean_absolute_error(y_test, y_test_pred))
valList.append(explained_variance_score(y_test, y_test_pred))
valList.append(1 - np.mean(np.abs(y_test_pred - y_test) / y_test))
bar.add('CNN-LSTM-ATTENTION测试集集评估指标', ['r^2', '均方差', '绝对差', '解释度', '平均绝对比准确率'],
        np.round(valList, 2), is_label_show=True, label_text_color='#000')
bar.render('CNN-LSTM-ATTENTION测试集评估指标.html')
print(trainList,valList)
# 4.可视化分析
plt.plot(np.array(y_test),
         color='darkorange', label='target')
plt.plot(y_test_pred,
         color='navy', label='predict')
plt.title("CNN-LSTM-ATTENTION results")
plt.legend()
plt.show()
# 5.保存模型
# m.save("./model_600.h5")
