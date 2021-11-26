import numpy as np


class LinearRegression:

    def fit(self, X, y):
        '''
        寻找到最优w*,记得处理特征量X以及标准化它
        '''
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def train(self, X, y):
        self.fit(X, y)

    def predict(self, X):
        '''
        返回(n,1)的矩阵,方便后面的计算
        '''
        return X.dot(self.w)

    def MSE(self, y_true, y_predict):
        '''
        计算均方误差*m
        '''
        #mean_squared_narray = (y_true-y_predict)**2
        #return np.average(mean_squared_narray, axis=0)
        # 求平均
        
        y_distance = y_true-y_predict
        return y_distance.T.dot(y_distance)[0,0]

    def R2_score(self, y_true, y_predict):
        '''
        计算决定系数
        '''
        #mean_squared_narray = (y_true-y_predict)**2
        #average_error = (y_true-np.average(y_true, axis=0))**2
        #return 1 - mean_squared_narray.sum(axis=0)/average_error.sum(axis=0)
        y_average = np.average(y_true, axis=0)
        y_distance = y_true-y_average
        return 1 - self.MSE(y_true,y_predict)/y_distance.T.dot(y_distance)[0,0]
