
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
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from data.dataset import dataset
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
# from pyecharts import Bar
def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')


# 1.数据预处理
data = pd.read_csv('../data/data.csv',
                   index_col=0, encoding='gb2312')
index = data.index
col = data.columns
class_names = np.unique(data.iloc[:, -1])
X = data.iloc[1:, ]
y = data.iloc[:-1, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=2020)

# 2.模型调参
# criterion = ['mse']
# n_estimators = [int(x) for x in np.linspace(start=400, stop=2000, num=5)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 100, num=5)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]
# random_grid = {'criterion': criterion,
#                'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# # 构建模型
# clf = RandomForestRegressor()
# clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
#                                 n_iter=10,
#                                 cv=10, verbose=2, random_state=2020, n_jobs=1)
# clf_random.fit(X, y)
# print(clf_random.best_params_)

# 3.训练模型
rf = RandomForestRegressor(criterion='mse', bootstrap=False, max_features='sqrt',
                           max_depth=10, min_samples_split=10, n_estimators=400, min_samples_leaf=1)
rf.fit(X_train, y_train)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# 3.验证评估
# print(rf.feature_importances_)
# bar1 = Bar()
# bar1.add('指标重要性', ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'AQI'], rf.feature_importances_.round(
#     2), is_label_show=True, label_text_color='#000')
# bar1.render('指标重要性.html')

# bar = Bar()
# trainList = []
# trainList.append(rf.score(X_train, y_train))
# trainList.append(mean_squared_error(y_train, y_train_pred))
# trainList.append(mean_absolute_error(y_train, y_train_pred))
# trainList.append(explained_variance_score(y_train, y_train_pred))
# trainList.append(1 - np.mean(np.abs(y_train_pred - y_train) / y_train))
# bar.add('RF训练集评估指标', ['r^2', '均方差', '绝对差', '解释度', '平均绝对比准确率'],
#         np.round(trainList, 2), is_label_show=True, label_text_color='#000')
# bar.render('RF训练集评估指标.html')

# valList = []
# valList.append(rf.score(X_test, y_test))
# valList.append(mean_squared_error(y_test, y_test_pred))
# valList.append(mean_absolute_error(y_test, y_test_pred))
# valList.append(explained_variance_score(y_test, y_test_pred))
# valList.append(1 - np.mean(np.abs(y_test_pred - y_test) / y_test))
# bar.add('RF测试集集评估指标', ['r^2', '均方差', '绝对差', '解释度', '平均绝对比准确率'],
#         np.round(valList, 2), is_label_show=True, label_text_color='#000')
# bar.render('RF测试集评估指标.html')

# 4.可视化分析
plt.plot(np.array(y_test),
         color='darkorange', label='target')
plt.plot(y_test_pred,
         color='navy', label='predict')
plt.title("随机森林预测结果",fontproperties=getChineseFont())
plt.legend()
plt.show()
