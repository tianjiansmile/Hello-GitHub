# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
from xgboost import plot_importance
from matplotlib import pyplot
import pandas as pd
from  com.risk_score import scorecard_functions_V3 as sf

# xgb 使用

# 用pandas 读数据
def pd_train():
    allData = pd.read_excel('pima.xlsx', sheetname='Sheet1')
    col = ['c1', 'c2', 'c3',
           'c4', 'c5', 'c6',
           'c7', 'c8']

    X = allData[col]
    Y = allData['y']

    trainData, testData = train_test_split(allData, test_size=0.4)

    X_train = trainData[col]
    Y_train = trainData['y']

    model = XGBClassifier()

    # 训练集，对没新加入的树进行测试
    eval_set = [(X_train, Y_train)]

    # model.fit(X_train, y_train)
    # early_stopping_rounds 连续10次loss不下降停止模型，
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    # make predictions for test data
    y_pred = model.predict(X_train)
    trainData['pred'] = y_pred

    # evaluate predictions
    accuracy = accuracy_score(Y_train, y_pred)
    print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

    print("AUC Score (Train): %f" % metrics.roc_auc_score(np.array(Y_train.T), y_pred))

    ks = sf.KS(trainData, 'pred', 'y')
    print('Test KS:', ks)

    # 测试集
    Y_test = testData['y']
    X_test = testData[col]
    y_test_pred = model.predict(X_test)
    testData['pred'] = y_test_pred

    accuracy = accuracy_score(Y_test, y_test_pred)
    print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

    print("Test AUC Score: %f" % metrics.roc_auc_score(np.array(Y_test.T), y_test_pred))

    ks = sf.KS(testData, 'pred', 'y')
    print('Test KS:', ks)

def train():
    # load data
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
    # split data into X and y
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # fit model no training data
    model = XGBClassifier()

    # 测试集，对没新加入的树进行测试
    eval_set = [(X_test, y_test)]

    # model.fit(X_train, y_train)
    # early_stopping_rounds 连续10次loss不下降停止模型，
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # traindata =
    # ks = sf.KS(trainData, 'pred', 'overdue_sum_label')

    print("AUC Score (Train): %f" % metrics.roc_auc_score(np.array(y_test.T), predictions))





# 调参
# 1.learning rate 学习率，就是梯度算法中的步长，太小导致收敛缓慢，太大导致无法收敛
# 2.tree 决策树的相关参数
# max_depth 最大深度
# min_child_weight 叶子节点的最小权重
# subsample, 建立树的样本数
# colsample_bytree
# gamma
# 3.正则化参数
# lambda
# alpha
def param_train():
    pass


if __name__ == '__main__':
    train()
    pd_train()
    # plot feature importance
    # plot_importance(model)
    # pyplot.show()