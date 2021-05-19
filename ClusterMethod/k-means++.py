"""
@File:k-means++.py
@Date:2021/5/18 19:53
@Author:博0_oer~
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


# 数据获取
def get_data():
    iris = load_iris()
    data = iris.data
    result = iris.target
    return data, result


# 为方便绘图，将数据集取前两个维度
def data_processing(data):
    data_list = []
    for i in data:
        tem_list = [i[2], i[3]]
        data_list.append(tem_list)
    return data_list


# 正确结果展示
def show_data(data, result):
    plt.title("right result")
    plt.xlabel("petal length")
    plt.ylabel("petal width")

    for i in range(len(result)):
        if result[i] == 0:
            plt.scatter(data[i][0], data[i][1], c="r")
        if result[i] == 1:
            plt.scatter(data[i][0], data[i][1], c="g")
        if result[i] == 2:
            plt.scatter(data[i][0], data[i][1], c="b")
    plt.show()


# 聚类迭代结果展示
def show_iter(class_result, new_center):
    plt.title("iter result")
    plt.xlabel("petal length")
    plt.ylabel("petal width")

    for cls in class_result:
        for c in cls:
            if class_result.index(cls) == 0:
                plt.scatter(c[0], c[1], c="r")
            if class_result.index(cls) == 1:
                plt.scatter(c[0], c[1], c="g")
            if class_result.index(cls) == 2:
                plt.scatter(c[0], c[1], c="b")
            if class_result.index(cls) == 3:
                plt.scatter(c[0], c[1], c="orange")
            if class_result.index(cls) == 4:
                plt.scatter(c[0], c[1], c="black")
            if class_result.index(cls) == 5:
                plt.scatter(c[0], c[1], c="yellow")
    for cen in new_center:
        if new_center.index(cen) == 0:
            plt.scatter(cen[0], cen[1], c="r", marker="v")
        if new_center.index(cen) == 1:
            plt.scatter(cen[0], cen[1], c="g", marker="v")
        if new_center.index(cen) == 2:
            plt.scatter(cen[0], cen[1], c="b", marker="v")
        if new_center.index(cen) == 3:
            plt.scatter(cen[0], cen[1], c="orange", marker="v")
        if new_center.index(cen) == 4:
            plt.scatter(cen[0], cen[1], c="black", marker="v")
        if new_center.index(cen) == 5:
            plt.scatter(cen[0], cen[1], c="yellow", marker="v")

    plt.show()


# k-means++类中心的选择
def select_centre(data, k):
    center_list = []
    ran = np.random.randint(len(data))
    center_list.append(data[ran])

    for i in range(k-1):
        # 构建数据点与中心点的距离
        dis_list = []
        for c in center_list:
            tmp_list = []
            for d in data:
                dis = np.sqrt(np.sum(np.square(np.array(d) - np.array(c))))
                tmp_list.append(dis)
            dis_list.append(tmp_list)
        # 计算每个中心点与其他点的距离和
        sum_dis_list = [0 for j in range(len(data))]
        for d in dis_list:
            sum_dis_list = np.array(sum_dis_list) + np.array(d)
        # 取距离和最大的点作为新的中心点
        center_list.append(data[list(sum_dis_list).index(max(sum_dis_list))])

    return center_list


# k-means聚类
def k_means(data, center):
    new_center = []
    cluster_result = []
    class_result = [[] for i in range(len(center))]
    for d in data:
        dis_list = []
        for c in center:
            # 欧氏距离进行距离度量
            dis = np.sqrt(np.sum(np.square(np.array(d)-np.array(c))))
            dis_list.append(dis)
        cluster_result.append(dis_list.index(min(dis_list)))
        class_result[dis_list.index(min(dis_list))].append(d)

    # 更新类中心
    for cls in class_result:
        x = 0
        y = 0
        for c in cls:
            x += c[0]
            y += c[1]
        if len(cls) == 0:
            new_center.append(center[class_result.index(cls)])
        else:
            new_center.append([round(x/len(cls), 4), round(y/len(cls), 4)])

    print(new_center)
    # print(len(class_result[0]), len(class_result[1]), len(class_result[2]))

    return cluster_result, class_result, new_center


# 迭代
def iter_(data, center, iter):
    cluster_result, class_result, new_center = k_means(data, center)
    for i in range(iter-1):
        cluster_result, class_result, new_center = k_means(data, new_center)
        show_iter(class_result, new_center)
    return cluster_result, class_result, new_center


# 引入兰德指数评价聚类结果
def metric(result, pred_result):
    print(metrics.adjusted_rand_score(result, pred_result))


if __name__ == '__main__':
    # 加载数据集
    datas, results = get_data()
    # 处理成二维形式
    datas = data_processing(datas)
    # 展示正确的分类结果
    show_data(datas, results)
    # 选择k-means的类中心点
    centers = select_centre(datas, 3)
    # 进行k-means聚类
    cluster_results, class_results, new_centers = iter_(datas, centers, 10)
    # print(cluster_results, new_centers)
    # show_iter(class_results, new_centers)
    # 评价指标
    metric(results, cluster_results)