import numpy as np


class LinearRegression:

    def train(self, X, y):
        '''
        寻找到最优w*,记得处理特征量X以及标准化它
        '''
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        '''
        返回(n,1)的矩阵,方便后面的计算
        '''
        return X.dot(self.w)

    def mean_squared_error(self, y_true, y_predict):
        '''
        计算均方误差
        '''
        mean_squared_narray = (y_true-y_predict)**2
        #求平均
        return np.average(mean_squared_narray, axis=0)

    def R2_score(self, y_true, y_predict):
        '''
        计算决定系数
        '''
        mean_squared_narray = (y_true-y_predict)**2
        average_error = (y_true-np.average(y_true, axis=0))**2
        return 1 - mean_squared_narray.sum(axis=0)/average_error.sum(axis=0)
