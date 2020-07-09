"""
Anthor:liu jia ming
Date:2020/7
Theme:AQI Prediction
"""
#-*- coding: utf-8 -*-
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.dataset import dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from matplotlib.font_manager import FontProperties
# from pyecharts import Bar
def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

# 1.数据预处理
data = pd.read_csv('../data/data.csv',
                   index_col=0, encoding='gb2312')
scaler = preprocessing.StandardScaler()
index = data.index
col = data.columns
class_names = np.unique(data.iloc[:, -1])
X = scaler.fit_transform(data.iloc[1:, ])
y = scaler.fit_transform(data.iloc[:-1, -1].values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=2020)

# 2.模型调参


# for kernel in ["linear", "poly", "rbf"]:
score = []

gamma_range = np.logspace(-10, 1, 50)
# c_range = np.linspace(0.01, 10, 50)
# for i in c_range:
#     svr = SVR(kernel='linear', C=i, cache_size=2000).fit(
#         X_train, y_train.ravel())
#     y_test_pred = svr.predict(X_test)
#     score.append(mean_squared_error(y_test_pred, y_test))
# print(min(score), c_range[score.index(min(score))])
# plt.plot(c_range, score)
# plt.show()
# c=0.8255

# for i in gamma_range:
#     svr = SVR(kernel='rbf', gamma=i, cache_size=2000).fit(
#         X_train, y_train.ravel())
#     y_test_pred = svr.predict(X_test)
#     score.append(mean_squared_error(y_test_pred, y_test))
# print(min(score), gamma_range[score.index(min(score))])
# plt.plot(gamma_range, score)
# plt.show()
# gamma = 0.007

# 3.训练模型
svr = SVR(kernel='linear', gamma=0.8255, cache_size=2000)
svr.fit(X_train, y_train)
y_train_pred = scaler.inverse_transform(svr.predict(X_train))
y_test_pred = scaler.inverse_transform(svr.predict(X_test))

# 4.验证评估


# bar = Bar()
# trainList = []
# trainList.append(svr.score(X_train, y_train))
# X_train = scaler.inverse_transform(X_train)
# y_train = scaler.inverse_transform(y_train)
# trainList.append(mean_squared_error(y_train, y_train_pred))
# trainList.append(mean_absolute_error(y_train, y_train_pred))
# trainList.append(explained_variance_score(y_train, y_train_pred))
# trainList.append(1 - np.mean(np.abs(y_train_pred - y_train) / y_train))
# bar.add('SVR训练集评估指标', ['r^2', '均方差', '绝对差', '解释度', '平均绝对比准确率'],
#         np.round(trainList, 2), is_label_show=True, label_text_color='#000')
# bar.render('SVR训练集评估指标.html')

# valList = []
# valList.append(svr.score(X_test, y_test))
# X_test = scaler.inverse_transform(X_test)
# y_test = scaler.inverse_transform(y_test)
# valList.append(mean_squared_error(y_test, y_test_pred))
# valList.append(mean_absolute_error(y_test, y_test_pred))
# valList.append(explained_variance_score(y_test, y_test_pred))
# valList.append(1 - np.mean(np.abs(y_test_pred - y_test) / y_test))
# bar.add('SVR测试集集评估指标', ['r^2', '均方差', '绝对差', '解释度', '平均绝对比准确率'],
#         np.round(valList, 2), is_label_show=True, label_text_color='#000')
# bar.render('SVR测试集评估指标.html')

# # 5.可视化分析
plt.plot(np.array(y_test),
         color='darkorange', label='target')
plt.plot(y_test_pred,
         color='navy', label='predict')
plt.title("支持向量机预测结果",fontproperties=getChineseFont())
plt.legend()
plt.show()
