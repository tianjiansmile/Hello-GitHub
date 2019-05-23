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

def label_map(x):
    if int(x) == 0:
        return x
    else:
        return 1

def train(trainData,testData,col):

    X_train = trainData[col]
    Y_train = trainData['overdue_days']

    # fit model no training data
    # model = XGBClassifier()

    model = XGBClassifier(
    learning_rate =0.2,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=6,
    seed=27)

    # 训练集，对没新加入的树进行测试
    eval_set = [(X_train, Y_train)]

    # early_stopping_rounds 连续10次loss不下降停止模型，
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    # make predictions for test data
    y_pred = model.predict(X_train)
    trainData['pred'] =y_pred
    y_predprob = model.predict_proba(X_train)[:, 1]

    print(trainData['overdue_days'].dtype,trainData['pred'].dtype)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(Y_train, predictions)
    print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

    # 注意ROC计算只针对二分类，确保参与计算的y值只有两种
    print("AUC Score (Train): %f" % metrics.roc_auc_score(trainData['overdue_days'], trainData['pred']))

    ks = sf.KS(trainData, 'pred', 'overdue_days')
    # ks = ks_calc_auc(trainData, 'pred', 'overdue_sum_label')
    print('Train KS:', ks)

    Y_test = testData['overdue_days']
    X_test = testData[col]
    y_test_pred = model.predict(X_test)

    test_predictions = [round(value) for value in y_test_pred]
    accuracy = accuracy_score(Y_test, test_predictions)
    print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

    testData['pred'] = y_test_pred

    # 注意ROC计算只针对二分类，确保参与计算的y值只有两种
    print("AUC Score (test): %f" % metrics.roc_auc_score(np.array(Y_test.T), y_test_pred))

    ks = sf.KS(testData, 'pred', 'overdue_days')
    print('Test KS:', ks)

    plot_importance(model)
    pyplot.show()

def miaola_test():
    allData = pd.read_excel('秒啦首贷_train_pd10.xlsx', sheetname='Sheet1')
    col = ['approve_rate_prov', 'approve_rate_city', 'approve_rate_country',
           'overdue_rate_prov', 'overdue_rate_city', 'overdue_rate_country',
           'avg_apply_prov', 'avg_apply_city', 'avg_apply_country',
           'avg_approve_prov', 'avg_approve_city', 'avg_approve_country',
           'avg_overdue_prov', 'avg_overdue_city', 'avg_overdue_country',
           'avg_loanamount_prov', 'avg_loanamount_city', 'avg_loanamount_country',

           'avg_pd3_prov', 'avg_pd3_city', 'avg_pd3_country',
           'avg_pd7_prov', 'avg_pd7_city', 'avg_pd7_country',
           'avg_pd10_prov', 'avg_pd10_city', 'avg_pd10_country',
           'avg_pd14_prov', 'avg_pd14_city', 'avg_pd14_country',
           'avg_M1_prov', 'avg_M1_city', 'avg_M1_country',
           'avg_M2_prov', 'avg_M2_city', 'avg_M2_country',
           'avg_M3_prov', 'avg_M3_city', 'avg_M3_country',

           'pd3_rate_prov', 'pd3_rate_city', 'pd3_rate_country',
           'pd7_rate_prov', 'pd7_rate_city', 'pd7_rate_country',
           'pd10_rate_prov', 'pd10_rate_city', 'pd10_rate_country',
           'pd14_rate_prov', 'pd14_rate_city', 'pd14_rate_country',
           'M1_rate_prov', 'M1_rate_city', 'M1_rate_country',
           'M2_rate_prov', 'M2_rate_city', 'M2_rate_country',
           'M3_rate_prov', 'M3_rate_city', 'M3_rate_country']

    # 暂时删除有空数据的行
    allData.dropna(axis=0, how='any', inplace=True)
    # 确保二分类
    allData['overdue_days'] = allData['overdue_days'].map(label_map)

    X = allData[col]

    trainData, testData = train_test_split(allData, test_size=0.33)

    # 训练
    train(trainData, testData,col)

if __name__ == '__main__':
    miaola_test()