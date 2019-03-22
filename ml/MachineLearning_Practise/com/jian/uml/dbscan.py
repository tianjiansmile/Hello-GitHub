from sklearn import metrics
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

if __name__ == '__main__':
    beer = pd.read_csv('data.txt', sep=' ')

    X = beer[["calories", "sodium", "alcohol", "cost"]]

    db = DBSCAN(eps=10, min_samples=2).fit(X)

    # 分类结果
    labels = db.labels_
    # print(labels)
    beer['cluster_db'] = labels
    beer.sort_values('cluster_db')

    # print(beer.sort_values('cluster_db'))
    #
    # print(beer.groupby('cluster_db').mean())

    colors = np.array(['red', 'green', 'blue', 'yellow'])
    scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10, 10), s=100)

    plt.show()
