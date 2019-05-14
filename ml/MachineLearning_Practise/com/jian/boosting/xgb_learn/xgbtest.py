# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot

# xgb 使用


# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()

def train():

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
    # plot feature importance
    # plot_importance(model)
    # pyplot.show()