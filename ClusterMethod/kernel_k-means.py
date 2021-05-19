"""
@File:kernel_k-means.py
@Date:2021/5/18 19:54
@Author:博0_oer~
"""
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import metrics


# 生成数据集
def make_data():
    return datasets.make_circles(n_samples=200, factor=.5, noise=.05)


# 数据集处理
def data_processing(data):
    data_ = list(data[0])
    data_list = []
    for d in data_:
        data_list.append(list(d))
    result = list(data[1])

    return data_list, result


# 数据集展示
def show_data(data, result):
    for d in data:
        if result[data.index(d)] == 0:
            plt.scatter(d[0], d[1], c="r")
        if result[data.index(d)] == 1:
            plt.scatter(d[0], d[1], c="g")
    plt.show()


# 将原始二维数据点映射到三维
# 引入一个新的维度，数据点与远点的距离，转换：（x，y）转换为（x，sqrt(x**2+y**2)，y）
def axes3d(data):
    new_data_list = []
    for d in data:
        new_data = [np.sqrt(d[0]**2+d[1]**2), d[0], d[1]]
        new_data_list.append(new_data)

    return new_data_list


# 三维图像绘制
def show_data_3d(data, result):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for d in data:
        if result[data.index(d)] == 0:
            ax.scatter(d[0], d[1], d[2], c="r", marker="8")
        if result[data.index(d)] == 1:
            ax.scatter(d[0], d[1], d[2], c="g", marker="8")
    plt.show()


if __name__ == '__main__':
    # 生成数据集
    datas = make_data()
    # print(datas)
    # 数据处理
    data, result = data_processing(datas)
    # print(data, result)
    # 展示数据正确的聚类图
    show_data(data, result)
    # 进行kernel坐标变换
    data_3d = axes3d(data)
    # print(data_3d)
    # 绘制三维图像
    show_data_3d(data_3d, result)


