import pandas as pd
import re
import time
import datetime
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from patsy.highlevel import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from numpy import log
from sklearn.metrics import roc_auc_score
import numpy as np
from  com.risk_score import scorecard_functions_V3 as sf

def random_forest_filter(train,X,y,var_WOE_list):
    RFC = RandomForestClassifier()
    RFC_Model = RFC.fit(X, y)
    features_rfc = train[var_WOE_list].columns
    featureImportance = {features_rfc[i]: RFC_Model.feature_importances_[i] for i in range(len(features_rfc))}
    featureImportanceSorted = sorted(featureImportance.items(), key=lambda x: x[1], reverse=True)
    # we selecte the top 10 features
    features_selection = [k[0] for k in featureImportanceSorted[:8]]

    y = train['y']
    X = train[features_selection]
    X['intercept'] = [1] * X.shape[0]

    LR = sm.Logit(y, X).fit()
    summary = LR.summary()

    print('RandomForest important featursorted', features_selection)

    return X