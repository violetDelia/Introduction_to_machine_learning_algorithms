import numpy as np
from LinearRegression import LinearRegression
import math


class RANSACLinearRegression(LinearRegression):

    def __init__(self, max_distance, qualification_rate=0.9, sub_rate=0.3, max_steps=20):
        '''
        max_distance: 允许的误差范围
        qualification_rate: 合格率
        sub_rate: 选取子集的比例
        max_steps: 迭代数
        '''
        self.max_distance = max_distance
        self.qualification_rate = qualification_rate
        self.sub_rate = sub_rate
        self.max_steps = max_steps

    def get_subset(self, Xy):
        m, n = Xy.shape
        np.random.shuffle(Xy)
        return Xy[:math.ceil(m*self.sub_rate), 0:n-1], Xy[:math.ceil(m*self.sub_rate), n-1:n]

    def check_passed(self, y_subset, y_subset_predict):
        m, n = y_subset.shape
        sum_count = m
        pass_count = 0
        for i in range(m):
            if(abs(y_subset[i]-y_subset_predict[i]) < self.max_distance):
                pass_count += 1
        return pass_count/sum_count > self.qualification_rate

    def train(self, X, y):
        step = 0
        min_MSE = float("inf")
        best_w = None
        Xy = np.concatenate((X, y), axis=1)
        while step < self.max_steps:
            X_subset, y_subset = self.get_subset(Xy)
            self.fit(X_subset, y_subset)
            y_subset_predict = self.predict(X_subset)
            if self.check_passed(y_subset, y_subset_predict):
                current_MSE = self.MSE(y_subset, y_subset_predict)
                if(current_MSE<min_MSE):
                    best_w = self.w
                    min_MSE = current_MSE
            step += 1
        self.w = best_w