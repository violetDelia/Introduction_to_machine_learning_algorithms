import numpy as np
from enum import Enum


class LinearRegressionMixin:
    '''
    基础的线性回归
    '''
    w = None

    def _normal(self, X, y):
        '''
        用正规方程求得w

        参数：
            X: 处理好的训练特征集
            y: 处理好的训练结果集
        '''
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def _fit_linear_normal(self, X_train, y_train):
        '''
        使用正则化方程直接求得最优解

        参数:
            X_train: 训练特征集
            y_train: 训练结果集
        '''
        self.w = self._normal(X_train, y_train)
