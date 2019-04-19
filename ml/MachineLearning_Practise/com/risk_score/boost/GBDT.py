import pandas as pd
import pickle
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import  metrics
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pylab as plt
import time
import datetime
from dateutil.relativedelta import relativedelta
from numpy import log
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model.logistic import LogisticRegression

# 将用户审核通过次数转换为是否审核通过
def map_label(x):
    if x == 0:
        return x
    else:
        return 1

if __name__ == '__main__':
    train = pd.read_excel('approve_feature.xls', sheetname='sheet1')

    # data_check(train)

    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    # 将通过次数转换为0,1标签
    # train['approve_sum_label'] = train['approve_sum_label'].map(map_label)

    # 将逾期次数转化为0，1标签
    train['overdue_sum_label'] = train['overdue_sum_label'].map(map_label)

    # 处理标签：Fully Paid是正常用户；Charged Off是违约用户
    train['y'] = train['overdue_sum_label']

    print(len(train['y'].unique()))


    # 将不参与训练的特征数据删除
    train.drop(['apply_int_label', 'apply_pdl_label', 'apply_sum_label'
                   , 'approve_int_label', 'approve_pdl_label', 'approve_sum_label', 'overdue_pdl_label',
                'overdue_int_label', 'overdue_sum_label', 'maxOverdue_pdl_label',
                'maxOverdue_int_label', 'maxOverdue_sum_label'], axis=1, inplace=True)

    trainData, testData = train_test_split(train, test_size=0.4)

    cat_features = [cont for cont in list(trainData.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['idNum,y']]

    # 变量类型超过5
    more_value_features = []
    less_value_features = []
    # 第一步，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
        valueCounts = len(set(trainData[var]))
        if valueCounts > 5:
            more_value_features.append(var)  # 取值超过5的变量，需要bad rate编码，再用卡方分箱法进行分箱
        else:
            less_value_features.append(var)

    print(more_value_features)
    print(less_value_features)

    v = DictVectorizer(sparse=False)
    X1 = v.fit_transform(trainData[less_value_features].to_dict('records'))
    # 将独热编码和数值型变量放在一起进行模型训练
    X2 = np.matrix(trainData[more_value_features])
    X = np.hstack([X1, X2])
    y = trainData['y']
    # 未经调参进行GBDT模型训练
    gbm0 = GradientBoostingClassifier(random_state=10)
    gbm0.fit(X, y)

    y_pred = gbm0.predict(X)
    y_predprob = gbm0.predict_proba(X)[:, 1].T
    print("Accuracy : %.4g" % metrics.accuracy_score(y, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(np.array(y.T), y_predprob))

