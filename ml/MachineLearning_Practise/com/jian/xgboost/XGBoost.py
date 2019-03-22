import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

# 首先，我们训练一个基本的xgboost模型，然后进行参数调节通过交叉验证来观察结果的变换，使用平均绝对误差来衡量
def simple_xgboost_train(train_x,label):
    # xgboost自定义了一个数据矩阵类
    # DMatrix，会在训练开始时进行一遍预处理，从而提高之后每次迭代的效率
    dtrain = xgb.DMatrix(train_x, label)

    # Xgboost参数
    # 'booster':'gbtree',
    # 'objective': 'multi:softmax', 多分类的问题
    # 'num_class':10, 类别数，与 multisoftmax 并用
    # 'gamma':损失下降多少才进行分裂
    # 'max_depth':12, 构建树的深度，越大越容易过拟合
    # 'lambda':2, 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    # 'subsample':0.7, 随机采样训练样本
    # 'colsample_bytree':0.7, 生成树时进行的列采样
    # 'min_child_weight':3, 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
    # 'silent':0 ,设置成1则没有运行信息输出，最好是设置为0.
    # 'eta': 0.007, 如同学习率
    # 'seed':1000,
    # 'nthread':7, cpu 线程数
    xgb_params = {
        'seed': 0,
        'eta': 0.1,
        'colsample_bytree': 0.5,
        'silent': 1,
        'subsample': 0.5,
        'objective': 'reg:linear',
        'max_depth': 5,
        'min_child_weight': 3
    }

    bst_cv1 = xgb.cv(xgb_params, dtrain, num_boost_round=50, nfold=3, seed=0,
                     feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)

    # 使用交叉验证 xgb.cv 我们得到了第一个基准结果：MAE＝1218.9
    print('CV score:', bst_cv1.iloc[-1, :]['test-mae-mean'])

    plt.figure()
    # 我们的第一个基础模型：没有发生过拟合 只建立了50个树模型
    bst_cv1[['train-mae-mean', 'test-mae-mean']].plot()
    plt.show()

    # 建立100个树模型
    bst_cv2 = xgb.cv(xgb_params, dtrain, num_boost_round=100,
                     nfold=3, seed=0, feval=xg_eval_mae, maximize=False,
                     early_stopping_rounds=10)

    print('CV score:', bst_cv2.iloc[-1, :]['test-mae-mean'])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16, 4)

    ax1.set_title('100 rounds of training')
    ax1.set_xlabel('Rounds')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.plot(bst_cv2[['train-mae-mean', 'test-mae-mean']])
    ax1.legend(['Training Loss', 'Test Loss'])

    ax2.set_title('60 last rounds of training')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.plot(bst_cv2.iloc[40:][['train-ma-mean', 'test-mae-mean']])
    ax2.legend(['Training Loss', 'Test Loss'])
    plt.show()


# 使用平均绝对误差来衡量
def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

# XGBoost 参数调节
# Step 1: 选择一组初始参数
# Step 2: 改变 max_depth 和 min_child_weight.
# Step 3: 调节 gamma 降低模型过拟合风险.
# Step 4: 调节 subsample 和 colsample_bytree 改变数据采样策略.
# Step 5: 调节学习率 eta.
class XGBoostRegressor(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
        self.params.update({'silent': 1, 'objective': 'reg:linear', 'seed': 0})

    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, y_train)
        self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                             feval=xg_eval_mae, maximize=False)

    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)

    def kfold(self, x_train, y_train, nfold=5):
        dtrain = xgb.DMatrix(x_train, y_train)
        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                           nfold=nfold, feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
        return cv_rounds.iloc[-1, :]

    def plot_feature_importances(self):
        feat_imp = pd.Series(self.bst.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self

def mae_score(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))

if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    # 做对数转换
    train['log_loss'] = np.log(train['loss'])

    # 提取参与运算的变量
    features = [x for x in train.columns if x not in ['id', 'loss', 'log_loss']]

    # 离散变量
    cat_features = [x for x in train.select_dtypes(
        include=['object']).columns if x not in ['id', 'loss', 'log_loss']]
    # 连续变量
    num_features = [x for x in train.select_dtypes(
        exclude=['object']).columns if x not in ['id', 'loss', 'log_loss']]

    print("Categorical features:", len(cat_features))
    print("Numerical features:", len(num_features))

    ntrain = train.shape[0]

    train_x = train[features]
    # 对数化之后的损失值用作标签
    train_y = train['log_loss']

    for c in range(len(cat_features)):
        train_x[cat_features[c]] = train_x[cat_features[c]].astype('category').cat.codes

    print("Xtrain:", train_x.shape)
    print("ytrain:", train_y.shape)

    simple_xgboost_train(train_x, train['log_loss'])

    mae_scorer = make_scorer(mae_score, greater_is_better=False)
    bst = XGBoostRegressor(eta=0.1, colsample_bytree=0.5, subsample=0.5,
                           max_depth=5, min_child_weight=3, num_boost_round=50)