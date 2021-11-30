import matplotlib.pyplot as plt
import numpy as np


class DataVisualization:
    '''
    数据可视化的类
    '''

    def plot_true_scatter_and_predict_scatter(self, X_feature, y_true, y_predict, column=0):
        '''
        显示真实值和预测值的散点图

        参数：
            X_feature: 特征数据集
            y_true: 真实的结果集
            y_predict: 预测的结果集
            colum: 要显示的特征列
        '''
        target_feature = X_feature[:, column]
        plt.scatter(target_feature, y_true)
        plt.scatter(target_feature, y_predict, color='red')
