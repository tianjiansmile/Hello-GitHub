# First XGBoost model for Pima Indians dataset
# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot
from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics
from  com.risk_score import scorecard_functions_V3 as sf
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

# xgb 使用

def label_map(x):
    if int(x) == 0:
        return x
    else:
        return 1

def plot_rel(col):
    # 特征之间的相关性
    plt.subplots(figsize=(16, 9))
    correlation_mat = col.corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()


def test():
    pass


def train(trainData,testData):

    X_train = trainData[col]
    # trainData['overdue_sum_label'] = trainData['overdue_sum_label'].map(num_map)
    Y_train = trainData['overdue_sum_label']

    # split data into train and test sets
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # fit model no training data
    model = XGBClassifier()

    # 训练集，对没新加入的树进行测试
    eval_set = [(X_train, Y_train)]

    # model.fit(X_train, y_train)
    # early_stopping_rounds 连续10次loss不下降停止模型，
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    # make predictions for test data
    y_pred = model.predict(X_train)
    trainData['pred'] =y_pred
    y_predprob = model.predict_proba(X_train)[:, 1]

    print(trainData['overdue_sum_label'].dtype,trainData['pred'].dtype)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y_train, predictions)
    print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

    # 注意ROC计算只针对二分类，确保参与计算的y值只有两种
    print("AUC Score (Train): %f" % metrics.roc_auc_score(trainData['overdue_sum_label'], trainData['pred']))

    ks = sf.KS(trainData, 'pred', 'overdue_sum_label')
    # ks = ks_calc_auc(trainData, 'pred', 'overdue_sum_label')
    print('Train KS:', ks)

    Y_test = testData['overdue_sum_label']
    X_test = testData[col]
    y_test_pred = model.predict(X_test)

    test_predictions = [round(value) for value in y_test_pred]
    accuracy = accuracy_score(Y_test, test_predictions)
    print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

    testData['pred'] = y_test_pred

    # 注意ROC计算只针对二分类，确保参与计算的y值只有两种
    print("AUC Score (test): %f" % metrics.roc_auc_score(np.array(Y_test.T), y_test_pred))

    ks = sf.KS(testData, 'pred', 'overdue_sum_label')
    print('Test KS:', ks)

    plot_importance(model)
    pyplot.show()

    # plot_rel(trainData)



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

def num_map(x):
    return int(x)


def ks_calc_auc(data,pred,y_label):
    '''
    功能: 计算KS值，输出对应分割点和累计分布函数曲线图
    输入值:
    data: 二维数组或dataframe，包括模型得分和真实的标签
    pred: 一维数组或series，代表模型得分（一般为预测正类的概率）
    y_label: 一维数组或series，代表真实的标签（{0,1}或{-1,1}）
    输出值:
    'ks': KS值
    '''
    fpr,tpr,thresholds= roc_curve(data[y_label[0]],data[pred[0]])
    ks = max(tpr-fpr)
    return ks

if __name__ == '__main__':
    allData = pd.read_excel('approve_addr_feature_train.xlsx', sheetname='Sheet1')
    col = ['approve_rate_prov', 'approve_rate_city', 'approve_rate_country',
           'overdue_rate_prov', 'overdue_rate_city', 'overdue_rate_country',
           'avg_loanamount_prov', 'avg_loanamount_city', 'avg_loanamount_country',
           'avg_apply_prov', 'avg_apply_city', 'avg_apply_country',
           'person_count_prov', 'person_count_city', 'person_count_country']

    # 暂时删除有空数据的行
    allData.dropna(axis=0, how='any', inplace=True)
    # 确保二分类
    allData['overdue_sum_label'] = allData['overdue_sum_label'].map(label_map)

    X = allData[col]
    Y = allData['overdue_sum_label']

    trainData, testData = train_test_split(allData, test_size=0.33)

    # 训练
    train(trainData,testData)