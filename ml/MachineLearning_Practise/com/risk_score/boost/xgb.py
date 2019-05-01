# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
from scipy.stats import ks_2samp


def fit_feat_map(df):
    array_columns = []
    feature_names = []
    transform = []
    for column in df.columns:
        column_series = df[column]
        if column_series.dtype.name == 'object' or column_series.dtype.name == 'category':
            labels, uniques = pd.factorize(column_series)
            dummies_map = {}
            for i in range(uniques.size):
                #print(uniques[i],column)
                dummies_map[uniques[i]] = i
                feature_names.append(column + '_' + uniques[i])
            transform.append((column, dummies_map))
            dummy_array = np.zeros((column_series.size, uniques.size))
            for i in range(column_series.size):
                #print(column_series.iloc[i],type(column_series.iloc[i]))
                if str(column_series.iloc[i]) == 'nan':
                    print(column, column_series.iloc[i])
                dummy_array[i, dummies_map[column_series.iloc[i]]] = 1
            array_columns.append(dummy_array)
        else:
            array_columns.append(column_series.values.reshape((column_series.size, 1)))
            transform.append((column, None))
            feature_names.append(column)
    return np.concatenate(array_columns, axis=1), transform, feature_names


def apply_feat_map(df, transform):
    array_column_list = []
    for column, dummies_map in transform:
        # print(dummies_map)
        column_series = df[column]
        if dummies_map is None:
            array_column_list.append(column_series.values.reshape((column_series.size, 1)))
        else:
            dummy_array = np.zeros((column_series.size, len(dummies_map)))
            for i in range(column_series.size):
                if column_series.iloc[i] in dummies_map:
                    dummy_array[i, dummies_map[column_series.iloc[i]]] = 1
            array_column_list.append(dummy_array)
    return np.concatenate(array_column_list, axis=1)


df = pd.read_excel(r'C:\Users\Administrator\Desktop\bjpost_20190428.xlsx', sheet_name='sheet2')

for i in range(1):
    cut = round(0.8 * df.shape[0])

    train_df0 = df.iloc[:cut]
    test_df0 = df.iloc[cut:]

    train_y_df0 = train_df0['label']
    train_x_df0 = train_df0.drop(labels=['cid', 'label'], axis=1)
    test_y_df0 = test_df0['label']
    test_x_df0 = test_df0.drop(labels=['cid', 'label'], axis=1)

    train_x_array, feat_map, feature_names = fit_feat_map(train_x_df0)

    train_y_array = train_y_df0.values
    test_x_array = apply_feat_map(test_x_df0, feat_map)
    test_y_array = test_y_df0.values

    dtrain = xgb.DMatrix(train_x_array, label=train_y_array, feature_names=feature_names)
    dtest = xgb.DMatrix(test_x_array, label=test_y_array, feature_names=feature_names)
    param = {'max_depth': 1,
             'eta': 0.01,
             'silent': 1,
             'objective': 'binary:logistic',
             'subsample': 0.5,
             'seed': datetime.datetime.now().microsecond
             }
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 800
    bst = xgb.train(param, dtrain, num_round, evallist)

    train_y_pred = bst.predict(dtrain)
    test_y_pred = bst.predict(dtest)

    train_ks = ks_2samp(train_y_pred[train_y_array == 1], train_y_pred[train_y_array == 0]).statistic
    test_ks = ks_2samp(test_y_pred[test_y_array == 1], test_y_pred[test_y_array == 0]).statistic

    print(train_ks)
    print(test_ks)


    feat_imp = bst.get_fscore()

file = open('0428_test_pd7.txt', 'w', encoding='utf-8')
f_i = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
for i in f_i:
    file.write(str(i[0]) + ':' + str(i[1]) + '\n')

file.close()