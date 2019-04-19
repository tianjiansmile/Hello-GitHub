# -*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

# 不同级包下的文件调用
from com.jian.logistic import logRegres

class matplot:
    def __init__(self):
        pass

    def drowMap(self,matr,classVector):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # 设置X轴标签
        plt.xlabel('X')
        # 设置Y轴标签
        plt.ylabel('Y')

        # 这里将喜欢不喜欢这一向量数据映射到了点的颜色上，用来区分不同情况
        # 最大的圈圈是黄色来表示的，看的出来
        # phone_gray_score 0，register_org_cnt 1，searched_org_cnt 2 behavior_check_score 3
        # contact_loan 4   call_count 5  called_count 6    做一个简单的数据分析
        # ax.scatter(matr[:, 0], matr[:, 1], 20.0 * array(classVector), 20.0 * array(classVector))
        # ax.scatter(matr[:, 0], matr[:, 5], 20.0 * array(classVector), 20.0 * array(classVector))
        ax.scatter(matr[:, 1], matr[:, 2], 20.0 * array(classVector), 20.0 * array(classVector))

        plt.show()


    #  画出3D图像
    def draw3DMap(self,matr,classVector):
        fig = plt.figure()
        ax = Axes3D(fig)
        #  分别设置三个坐标轴代表的数据，把数据的类别 1 2 3 映射到散点数据点的大小size，和 数据点的颜色
        # phone_gray_score 0，register_org_cnt 1，searched_org_cnt 2 behavior_check_score 3
        # contact_loan 4   call_count 5  called_count 6

        # plt.xlim()
        # ax.scatter(matr[:, 0], matr[:, 1], matr[:, 4], matr[:, 2], 8.0 * array(classVector), 8.0 * array(classVector), depthshade=True)
        plt.ylim(0, 50)
        ax.scatter(matr[:, 0], matr[:, 1], matr[:, 2], matr[:, 2], 10.0 * array(classVector), 10.0 * array(classVector), depthshade=True)
        # ax.scatter(matr[:, 0], matr[:, 5], matr[:, 6], matr[:, 2], 10.0 * array(classVector), 10.0 * array(classVector), depthshade=True)
        # ax.scatter(matr[:, 4], matr[:, 5], matr[:, 6], matr[:, 2], 15.0 * array(classVector), 15.0 * array(classVector), depthshade=True)

        # plt.xlim(0,50);plt.ylim(0,50)
        # ax.scatter(matr[:, 1], matr[:, 5], matr[:, 6], matr[:, 2], 15.0 * array(classVector), 15.0 * array(classVector), depthshade=True)

        plt.show()


    # 根据特征画出折线图
    def drawLineMap(self,matr):
        x = linspace(0, len(matr), len(matr))
        # 绘制y=2x+1函数的图像
        y1 = matr[:, 0]
        y2 = matr[:, 1]
        plt.title('Result Analysis')
        plt.plot(x, y1, color='green', label='order')
        plt.plot(x, y2, color='skyblue', label='repay')

        plt.xlabel('date ')
        plt.ylabel('data different')
        plt.show()
        # python 一个折线图绘制多个曲线

    def type_check(self,item):
        try:
            data_type = 'list'
            if type(item).__name__ == 'list':
                data_type = 'list'
            elif type(item).__name__ == 'dict':
                data_type = 'dict'
            elif type(item).__name__ == 'str':
                data_type = 'str'
            elif type(item).__name__ == 'int':
                data_type = 'int'
            else:
                data_type = 'undefined'

            return data_type
        except:
            print("数据类型判断异常,%s" % str(item))

    # 转化数据标签从0 1 到其他值，
    def tranfrom(self,labelMat):
        data_type = self.type_check(labelMat)
        if data_type == 'list':
            field_type = self.type_check(labelMat[0])
            if field_type == 'int':
                count = 0
                for label in labelMat:
                    if label == 0:
                        labelMat[count] = 1
                    else:
                        labelMat[count] = 3
                    count +=1


            elif field_type == 'str':
                count = 0
                for label in labelMat:
                    if int(label) == 0:
                        labelMat[count] = 1
                    else:
                        labelMat[count] = 3
                    count += 1
            else:
                print(field_type)

        return labelMat

if __name__ == '__main__':
    mat_plot = matplot()

    dataMat,labelMat = logRegres.loadDataSet()

    data_matr = array(dataMat)
    # print(shape(data_matr))
    labelMat = mat_plot.tranfrom(labelMat)

    # print(data_matr,labelMat)

    mat_plot.draw3DMap(data_matr,labelMat)
