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
from com.risk_score import feature_filter

# 数据的标签，拒绝， 通过（通过贷后正常，通过贷后逾期），
# 按照通过还是拒绝来预测是否通过
# 以是否通过为数据标签得到一大批特征的VI值在0.02以上 是有效的

# 按照贷后来预测贷后逾期情况
# 目前来看数据样本不平衡，申请通过的订单很少，而申请通过的订单中逾期情况更少
# 如果要预测贷后是否逾期，样本的预备应该是所有申请通过的订单分为逾期和不逾期的

# 本程序旨在筛选金盘特征中比较有效的特征，之前已经对老的历史特征做过评估，
# 这一次我通过追加不同类型的特征来对比前后的模型表现效果
# 1 老特征
# 2 加入比率特征
# 3 加入网络特征



def data_check(train):
    # print('First 20 columns:', list(train.columns[:20]))
    print(train.shape)
    print(train.describe())

    # 查看空值情况 ,数据已经被清洗过，非常clean
    print(pd.isnull(train).values.any())

    # 数据的整体状况
    print(train.info())

    # 检查空值比例
    check_null = train.isnull().sum(axis=0).sort_values(ascending=False)
    print(check_null[check_null > 0])

    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    print(pd.isnull(train).values.any())

    cont_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['idNum']]
    # print("Continuous: {} features".format(len(cont_features)))

    # 统计离散变量的类别数
    cat_uniques = []
    for cat in cont_features:
        cat_uniques.append(len(train[cat].unique()))

    uniq_values_in_categories = pd.DataFrame.from_items([('cat_name', cont_features), ('unique_values', cat_uniques)])
    print(uniq_values_in_categories)



    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(16, 5)
    # ax1.hist(uniq_values_in_categories.unique_values, bins=50)
    # ax1.set_title('Amount of categorical features with X distinct values')
    # ax1.set_xlabel('Distinct values in a feature')
    # ax1.set_ylabel('Features')
    # ax1.annotate('A feature with 326 vals', xy=(322, 2), xytext=(200, 38), arrowprops=dict(facecolor='black'))
    #
    # ax2.set_xlim(2, 30)
    # ax2.set_title('Zooming in the [0,30] part of left histogram')
    # ax2.set_xlabel('Distinct values in a feature')
    # ax2.set_ylabel('Features')
    # ax2.grid(True)
    # ax2.hist(uniq_values_in_categories[uniq_values_in_categories.unique_values <= 30].unique_values, bins=30)
    # ax2.annotate('Binary features', xy=(3, 71), xytext=(7, 71), arrowprops=dict(facecolor='black'))

    # 特征之间的相关性
    # plt.subplots(figsize=(16, 9))
    # correlation_mat = train[cont_features].corr()
    # sns.heatmap(correlation_mat, annot=True)
    # plt.show()

# 将用户审核通过次数转换为是否审核通过
def map_label(x):
    if x == 0:
        return x
    else:
        return 1
'''
第三步：分箱，采用ChiMerge,要求分箱完之后：
（1）不超过5箱
（2）Bad Rate单调
（3）每箱同时包含好坏样本
（4）特殊值如－1，单独成一箱

连续型变量可直接分箱
类别型变量：
（a）当取值较多时，先用bad rate编码，再用连续型分箱的方式进行分箱
（b）当取值较少时：
    （b1）如果每种类别同时包含好坏样本，无需分箱
    （b2）如果有类别只包含好坏样本的一种，需要合并
'''
def box_split(train):

    cat_features = [cont for cont in list(train.select_dtypes(
        include=['float64', 'int64']).columns) if cont not in ['idNum,y']]

    # 变量类型超过5
    more_value_features = []
    less_value_features = []
    # 第一步，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
        valueCounts = len(set(train[var]))
        if valueCounts > 5:
            more_value_features.append(var)  # 取值超过5的变量，需要bad rate编码，再用卡方分箱法进行分箱
        else:
            less_value_features.append(var)


    print(more_value_features)
    print(less_value_features)

    # （i）当取值<5时：如果每种类别同时包含好坏样本，无需分箱；如果有类别只包含好坏样本的一种，需要合并
    merge_bin_dict = {}  # 存放需要合并的变量，以及合并方法
    var_bin_list = []  # 由于某个取值没有好或者坏样本而需要合并的变量
    for col in less_value_features:
        #  bad rate ,某一个变量对应所有的标签必须涵盖两种标签，也就是单调
        binBadRate = sf.BinBadRate(train, col, 'y')[0]
        if min(binBadRate.values()) == 0:  # 由于某个取值没有坏样本而进行合并
            print('{} need to be combined due to 0 bad rate'.format(col))
            combine_bin = sf.MergeBad0(train, col, 'y')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            train[newVar] = train[col].map(combine_bin)
            var_bin_list.append(newVar)
        if max(binBadRate.values()) == 1:  # 由于某个取值没有好样本而进行合并
            print('{} need to be combined due to 0 good rate'.format(col))
            combine_bin = sf.MergeBad0(train, col, 'y', direction='good')
            merge_bin_dict[col] = combine_bin
            newVar = col + '_Bin'
            train[newVar] = train[col].map(combine_bin)
            var_bin_list.append(newVar)

    # 保存merge_bin_dict
    # with open('merge_bin_dict.pkl','wb+') as wf:
    #     pickle.dump(merge_bin_dict, wf)

    # less_value_features里剩下不需要合并的变量
    less_value_features = [i for i in less_value_features if i + '_Bin' not in var_bin_list]

    # 连续变量
    num_features = []
    # （ii）当取值>5时：用bad rate进行编码，放入连续型变量里
    br_encoding_dict = {}  # 记录按照bad rate进行编码的变量，及编码方式
    for col in more_value_features:
        br_encoding = sf.BadRateEncoding(train, col, 'y')
        train[col + '_br_encoding'] = br_encoding['encoding']
        br_encoding_dict[col] = br_encoding['bad_rate']
        num_features.append(col + '_br_encoding')

    # 保存 br_encoding_dict
    # with open('br_encoding_dict.pkl','wb+') as wf:
    #     pickle.dump(br_encoding_dict, wf)

    # （iii）对连续型变量进行分箱，包括（ii）中的变量
    continous_merged_dict = {}
    for col in num_features:
        print("{} is in processing".format(col))
        if -1 not in set(train[col]):  # －1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
            max_interval = 5  # 分箱后的最多的箱数
            cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[], minBinPcnt=0)
            train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
            monotone = sf.BadRateMonotone(train, col + '_Bin', 'y')  # 检验分箱后的单调性是否满足
            while (not monotone):
                # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
                max_interval -= 1
                cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[],
                                  minBinPcnt=0)
                train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
                if max_interval == 2:
                    # 当分箱数为2时，必然单调
                    break
                monotone = sf.BadRateMonotone(train, col + '_Bin', 'y')
            newVar = col + '_Bin'
            train[newVar] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[]))
            var_bin_list.append(newVar)
        else:
            max_interval = 5
            # 如果有－1，则除去－1后，其他取值参与分箱
            cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[-1],
                              minBinPcnt=0)
            train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
            monotone = sf.BadRateMonotone(train, col + '_Bin', 'y', ['Bin -1'])
            while (not monotone):
                max_interval -= 1
                # 如果有－1，－1的bad rate不参与单调性检验
                cutOff = sf.ChiMerge(train, col, 'y', max_interval=max_interval, special_attribute=[-1],
                                  minBinPcnt=0)
                train[col + '_Bin'] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
                if max_interval == 3:
                    # 当分箱数为3-1=2时，必然单调
                    break
                monotone = sf.BadRateMonotone(train, col + '_Bin', 'y', ['Bin -1'])
            newVar = col + '_Bin'
            train[newVar] = train[col].map(lambda x: sf.AssignBin(x, cutOff, special_attribute=[-1]))
            var_bin_list.append(newVar)
        continous_merged_dict[col] = cutOff

    # 保存 continous_merged_dict
    # with open('continous_merged_dict.pkl', 'wb+') as wf:
    #     pickle.dump(continous_merged_dict, wf)

    '''
    第四步：WOE编码、计算IV
    '''
    WOE_dict = {}
    IV_dict = {}
    # 分箱后的变量进行编码，包括：
    # 1，初始取值个数小于5，且不需要合并的类别型变量。存放在less_value_features中
    # 2，初始取值个数小于5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
    # 3，初始取值个数超过5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
    # 4，连续变量。分箱后新的变量存放在var_bin_list中
    all_var = var_bin_list + less_value_features
    for var in all_var:
        woe_iv = sf.CalcWOE(train, var, 'y')
        WOE_dict[var] = woe_iv['WOE']
        IV_dict[var] = woe_iv['IV']

    # 将变量IV值进行降序排列，方便后续挑选变量
    IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)

    IV_values = [i[1] for i in IV_dict_sorted]
    IV_name = [i[0] for i in IV_dict_sorted]
    plt.title('feature IV')
    plt.bar(range(len(IV_values)), IV_values)

    print('IV sort',IV_values)
    print('IV_name', IV_name)

    '''
    第五步：单变量分析和多变量分析，均基于WOE编码后的值。
    （1）选择IV高于0.01的变量
    （2）比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
    '''

    IV_dict.pop('loanamount_pdl_7_br_encoding_Bin')

    IV_dict.pop('loanamount_int_7_br_encoding_Bin')
    IV_dict.pop('loan_avg_pdl_7_br_encoding_Bin')
    #
    IV_dict.pop('loan_avg_int_7_br_encoding_Bin')
    IV_dict.pop('loanamount_pdl_14_br_encoding_Bin')
    # IV_dict.pop('apply_int_diff_11_br_encoding_Bin')
    # IV_dict.pop('approve_mert_pdl_diff_3_br_encoding_Bin')

    # 选取IV>0.01的变量
    high_IV = {k: v for k, v in IV_dict.items() if v >= 0.02}
    high_IV_sorted = sorted(high_IV.items(), key=lambda x: x[1], reverse=True)

    short_list = high_IV.keys()
    short_list_2 = []
    for var in short_list:
        newVar = var + '_WOE'
        train[newVar] = train[var].map(WOE_dict[var])
        short_list_2.append(newVar)

    # 对于上一步的结果，计算相关系数矩阵，并画出热力图进行数据可视化
    trainDataWOE = train[short_list_2]
    # f, ax = plt.subplots(figsize=(10, 8))
    # corr = trainDataWOE.corr()
    # sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
    #             square=True, ax=ax)

    plt.show()

    # 两两间的线性相关性检验
    # 1，将候选变量按照IV进行降序排列
    # 2，计算第i和第i+1的变量的线性相关系数
    # 3，对于系数超过阈值的两个变量，剔除IV较低的一个
    deleted_index = []
    cnt_vars = len(high_IV_sorted)
    for i in range(cnt_vars):
        if i in deleted_index:
            continue
        x1 = high_IV_sorted[i][0] + "_WOE"
        for j in range(cnt_vars):
            if i == j or j in deleted_index:
                continue
            y1 = high_IV_sorted[j][0] + "_WOE"
            roh = np.corrcoef(train[x1], train[y1])[0, 1]
            if abs(roh) > 0.7:
                x1_IV = high_IV_sorted[i][1]
                y1_IV = high_IV_sorted[j][1]
                if x1_IV > y1_IV:
                    deleted_index.append(j)
                else:
                    deleted_index.append(i)

    multi_analysis_vars_1 = [high_IV_sorted[i][0] + "_WOE" for i in range(cnt_vars) if i not in deleted_index]

    '''
    多变量分析：VIF
    '''
    X = np.matrix(train[multi_analysis_vars_1])
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    max_VIF = max(VIF_list)
    print('多变量分析 maxVIF',max_VIF)
    # 最大的VIF是1.32267733123，因此这一步认为没有多重共线性
    multi_analysis = multi_analysis_vars_1



    '''
    第六步：逻辑回归模型。
    要求：
    1，变量显著
    2，符号为负
    '''
    ### (1)将多变量分析的后变量带入LR模型中
    y = train['y']
    X = train[multi_analysis]
    X['intercept'] = [1] * X.shape[0]

    LR = sm.Logit(y, X).fit()
    summary = LR.summary()
    pvals = LR.pvalues
    pvals = pvals.to_dict()

    # ### 有些变量不显著，需要逐步剔除
    varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
    varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)
    while (len(varLargeP) > 0 and len(multi_analysis) > 0):
        # 每次迭代中，剔除最不显著的变量，直到
        # (1) 剩余所有变量均显著
        # (2) 没有特征可选
        varMaxP = varLargeP[0][0]
        print(varMaxP)
        if varMaxP == 'intercept':
            print('the intercept is not significant!')
            break
        multi_analysis.remove(varMaxP)
        y = train['y']
        X = train[multi_analysis]
        X['intercept'] = [1] * X.shape[0]

        LR = sm.Logit(y, X).fit()
        pvals = LR.pvalues
        pvals = pvals.to_dict()
        varLargeP = {k: v for k, v in pvals.items() if v >= 0.1}
        varLargeP = sorted(varLargeP.items(), key=lambda d: d[1], reverse=True)

    summary = LR.summary()
    print(summary)

    train['pred'] = LR.predict(X)
    ks = sf.KS(train, 'pred', 'y')
    # ks = sf.ks_calc_auc(train,train['pred'],train['y'])
    auc = roc_auc_score(train['y'], train['pred'])  # AUC = 0.73
    print('准确度Area Under Curve auc',auc,'区分度 KS',ks)

    #############################################################################################################
    # 尝试用L1约束#
    ## use cross validation to select the best regularization parameter
    # multi_analysis = multi_analysis_vars_1
    # X = train[multi_analysis]  # by default  LogisticRegressionCV() fill fit the intercept
    # X = np.matrix(X)
    # y = train['approve_sum_label']
    # y = np.array(y)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    # X_train.shape, y_train.shape
    # #
    # model_parameter = {}
    # for C_penalty in np.arange(0.005, 0.2, 0.005):
    #     for bad_weight in range(2, 101, 2):
    #         print
    #         C_penalty, bad_weight
    #         LR_model_2 = LogisticRegressionCV(Cs=[C_penalty], penalty='l1', solver='liblinear',
    #                                           class_weight={1: bad_weight, 0: 1})
    #         LR_model_2_fit = LR_model_2.fit(X_train, y_train)
    #         y_pred = LR_model_2_fit.predict_proba(X_test)[:, 1]
    #         scorecard_result = pd.DataFrame({'prob': y_pred, 'target': y_test})
    #         performance = sf.KS_AR(scorecard_result,'prob','target')
    #         KS = performance['KS']
    #         model_parameter[(C_penalty, bad_weight)] = KS

    # 用随机森林法估计变量重要性#
    #
    var_WOE_list = multi_analysis_vars_1
    X = train[var_WOE_list]
    X = np.matrix(X)
    y = train['y']
    y = np.array(y)

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

    train['pred'] = LR.predict(X)
    ks = sf.KS(train, 'pred', 'y')
    # ks = sf.ks_calc_auc(train,train['pred'],train['y'])
    auc = roc_auc_score(train['y'], train['pred'])  # AUC = 0.73
    print('准确度Area Under Curve auc', auc, '区分度 KS', ks)

    # 用GBDT跑出变量重要性，挑选出合适的变量
    clf = ensemble.GradientBoostingClassifier()
    gbdt_model = clf.fit(X, y)
    importace = gbdt_model.feature_importances_.tolist()
    featureImportance = zip(multi_analysis, importace)
    featureImportanceSorted = sorted(featureImportance, key=lambda k: k[1], reverse=True)

    print('GBDT important featursorted', featureImportanceSorted)


# 测试是否通过
# feature.xls 是老的历史特征
# feature_pro.xls 添加了新的比率以及网络特征，我们将对比网络特征加入前后的模型表现
def approve_predict(file):
    train = pd.read_excel(file, sheetname='sheet1')

    # data_check(train)

    # 删除任何一行有空值的记录
    train.dropna(axis=0, how='any', inplace=True)

    # 将通过次数转换为0,1标签
    train['approve_sum_label'] = train['approve_sum_label'].map(map_label)

    # 将逾期次数转化为0，1标签
    # train['overdue_sum_label'] = train['overdue_sum_label'].map(map_label)

    # 处理标签：Fully Paid是正常用户；Charged Off是违约用户
    train['y'] = train['approve_sum_label']

    print(len(train['y'].unique()))

    # 将不参与训练的特征数据删除
    train.drop(['apply_int_label', 'apply_pdl_label', 'apply_sum_label'
                   , 'approve_int_label', 'approve_pdl_label', 'approve_sum_label', 'overdue_pdl_label',
                'overdue_int_label', 'overdue_sum_label', 'maxOverdue_pdl_label',
                'maxOverdue_int_label', 'maxOverdue_sum_label'], axis=1, inplace=True)

    # 删除rate特征
    # feature_control_drop(train)
    # # 删除network特征
    # feature_call_drop(train)
    # feature_contact_drop(train)

    # 删除迁移特征
    feature_mer_drop(train)

    box_split(train)

def overdue_predict(file):

    train = pd.read_excel(file, sheetname='sheet1')

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

    # feature_control_drop(train)
    # feature_call_drop(train)
    # feature_contact_drop(train)

    box_split(train)

# 删除比率特征
def feature_control_drop(train):
    # 将不参与训练的特征数据删除
    col_time = ['7','14','30','60','90','180','all']
    col_basic = ['approve_rate_pdl_7', 'approve_rate_int_7', 'approve_rate_sum_7'
                , 'reject_rate_pdl_7', 'reject_rate_int_7', 'reject_rate_sum_7',
                'loan_rate_pdl_7','loan_rate_int_7', 'loan_rate_sum_7',
                'overdue_rate_pdl_7','overdue_rate_int_7', 'overdue_rate_sum_7',
                'loan_avg_pdl_7','loan_avg_int_7','loan_avg_sum_7',
                'loan_times_pdl_7','loan_times_int_7','loan_times_sum_7'
                ]
    col_drop = []
    for i in range(len(col_time)-1):
        for j in range(len(col_basic)):
            col_drop.append(col_basic[j])
            col_basic[j] = col_basic[j].replace(col_time[i],col_time[i+1])

    for col in col_basic:
        col_drop.append(col)

    # print(col_drop)
    train.drop(col_drop, axis=1, inplace=True)

# 删除网络特征
def feature_call_drop(train):
    col_drop = ['call_max_overdue_sum','call_max_overdue_pdl','call_max_overdue_int',
                'call_max_overdue_times','call_max_apply_sum','call_max_approve_sum',
                'call_max_loanamount_sum','call_avg_overdue_sum','call_avg_overdue_pdl',
                'call_avg_overdue_int','call_avg_overdue_times','call_avg_apply_sum',
                'call_avg_approve_sum','call_avg_loanamount_sum']

    train.drop(col_drop, axis=1, inplace=True)

def feature_contact_drop(train):
    col_drop = ['contact_max_overdue_sum','contact_max_overdue_pdl','contact_max_overdue_int',
                'contact_max_overdue_times','contact_max_apply_sum','contact_max_approve_sum',
                'contact_max_loanamount_sum','contact_avg_overdue_sum','contact_avg_overdue_pdl',
                'contact_avg_overdue_int','contact_avg_overdue_times','contact_avg_apply_sum',
                'contact_avg_approve_sum','contact_avg_loanamount_sum']

    train.drop(col_drop, axis=1, inplace=True)

def feature_mer_drop(train):
    col_drop = ['apply_pdl_diff_1','apply_int_diff_1','approve_pdl_diff_1','approve_int_diff_1',
                 'apply_pdl_diff_2', 'apply_int_diff_2', 'approve_pdl_diff_2', 'approve_int_diff_2',
                 'apply_pdl_diff_3', 'apply_int_diff_3', 'approve_pdl_diff_3', 'approve_int_diff_3',
                 'apply_pdl_diff_4', 'apply_int_diff_4', 'approve_pdl_diff_4', 'approve_int_diff_4',
                 'apply_pdl_diff_5', 'apply_int_diff_5', 'approve_pdl_diff_5', 'approve_int_diff_5',
                 'apply_pdl_diff_6', 'apply_int_diff_6', 'approve_pdl_diff_6', 'approve_int_diff_6',
                 'apply_pdl_diff_7', 'apply_int_diff_7', 'approve_pdl_diff_7', 'approve_int_diff_7',
                 'apply_pdl_diff_8', 'apply_int_diff_8', 'approve_pdl_diff_8', 'approve_int_diff_8',
                 'apply_pdl_diff_9', 'apply_int_diff_9', 'approve_pdl_diff_9', 'approve_int_diff_9',
                 'apply_pdl_diff_10', 'apply_int_diff_10', 'approve_pdl_diff_10', 'approve_int_diff_10',
                 'apply_pdl_diff_11', 'apply_int_diff_11', 'approve_pdl_diff_11', 'approve_int_diff_11',
                 'apply_pdl_diff_12', 'apply_int_diff_12', 'approve_pdl_diff_12', 'approve_int_diff_12',

                 'apply_mert_pdl_diff_1', 'apply_mert_int_diff_1', 'approve_mert_pdl_diff_1', 'approve_mert_int_diff_1',
                 'apply_mert_pdl_diff_2', 'apply_mert_int_diff_2', 'approve_mert_pdl_diff_2','approve_mert_int_diff_2',
                 'apply_mert_pdl_diff_3', 'apply_mert_int_diff_3', 'approve_mert_pdl_diff_3','approve_mert_int_diff_3',
                 'apply_mert_pdl_diff_4', 'apply_mert_int_diff_4', 'approve_mert_pdl_diff_4', 'approve_mert_int_diff_4',
                 'apply_mert_pdl_diff_5', 'apply_mert_int_diff_5', 'approve_mert_pdl_diff_5','approve_mert_int_diff_5',
                 'apply_mert_pdl_diff_6', 'apply_mert_int_diff_6', 'approve_mert_pdl_diff_6','approve_mert_int_diff_6',
                 'apply_mert_pdl_diff_7', 'apply_mert_int_diff_7', 'approve_mert_pdl_diff_7','approve_mert_int_diff_7',
                 'apply_mert_pdl_diff_8', 'apply_mert_int_diff_8', 'approve_mert_pdl_diff_8','approve_mert_int_diff_8',
                 'apply_mert_pdl_diff_9', 'apply_mert_int_diff_9', 'approve_mert_pdl_diff_9','approve_mert_int_diff_9',
                 'apply_mert_pdl_diff_10', 'apply_mert_int_diff_10', 'approve_mert_pdl_diff_10','approve_mert_int_diff_10',
                 'apply_mert_pdl_diff_11', 'apply_mert_int_diff_11', 'approve_mert_pdl_diff_11','approve_mert_int_diff_11',
                 'apply_mert_pdl_diff_12', 'apply_mert_int_diff_12', 'approve_mert_pdl_diff_12','approve_mert_int_diff_12',
                 'apply_mert_pdl_sum', 'apply_mert_int_sum', 'approve_mert_pdl_sum','approve_mert_int_sum']

    train.drop(col_drop, axis=1, inplace=True)

if __name__ == '__main__':
    starttime = time.time()
    # 是否通过预测
    # file = 'feature_pro.xlsx'
    # approve_predict(file)

    # file = 'approve_feature_pro.xlsx'
    # overdue_predict(file)

    # 新的一批样本，全部为通过样本 样本量46000，逾期用户4400，基本为paydayloan
    # file = 'new_approve_feature _clean.xlsx'
    # overdue_predict(file)

    # 重新计算了逾期情况 样本量46000，逾期用户8000多，基本为paydayloan
    file = 'new_approve_feature _clean1.xlsx'
    overdue_predict(file)

    # 新的一批样本，全部为通过样本 样本量85000，通过用户64000，为paydayloan
    # file = 'zp_0425_SZZN_payday_feature.xlsx'
    # approve_predict(file)

    endtime = time.time()
    print(' cost time: ', endtime - starttime)