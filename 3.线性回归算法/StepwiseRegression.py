import numpy as np
from scipy.stats import f
from LinearRegression import LinearRegression
from enum import Enum


class StepwiseRegression(LinearRegression):

    class StepwiseRegressionType(Enum):
        forward_selection = 1
        backward_selection = 2

    def fit(self, X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def forward_f_test(self, MSE_A, MSE_min, m, confidence_interval=0.95):
        '''
        F检验，判断MSE_min小于MSE_A
        confidence_interval 置信度
        '''
        if MSE_min > MSE_A:
            return False
        F = MSE_A/MSE_min
        p_value = f.cdf(F, m, m)
        return p_value > confidence_interval

    def backward_f_test(self, MSE_A, MSE_min, m, confidence_interval=0.95):
        '''
        向后逐步的F检验
        '''
        if MSE_min < MSE_A:
            return True
        F = MSE_min/MSE_A
        p_value = f.cdf(F, m, m)
        return p_value <= confidence_interval

    def fit_and_compute_mse(self, X, y):
        w = self.fit(X, y)
        y_distances = y - X.dot(w)
        return y_distances.T.dot(y_distances)[0, 0]

    def forward_selection(self, X, y,confidence_interval=0.95):
        '''
        向前逐步回归
        '''
        m, n = X.shape
        A, C = [0], [i for i in range(1, n)]
        while len(C) > 0:
            # A集合的均方误差
            MSE_A = self.fit_and_compute_mse(X[:, A], y)
            MSE_min = float("inf")
            j_min = -1
            for j in C:
                # 加入j特征列后的均方误差
                MSE_j = self.fit_and_compute_mse(X[:, A+[j]], y)
                if MSE_j < MSE_min:
                    MSE_min, j_min = MSE_j, j
            # F检验
            if self.forward_f_test(MSE_A, MSE_min, m,confidence_interval=confidence_interval):
                A.append(j_min)
                C.remove(j_min)
            else:
                break
        self.A = A

    def backward_selection(self, X, y,confidence_interval=0.95):
        '''
        向后逐步回归
        '''
        m, n = X.shape
        A, C = [i for i in range(0, n)], []
        while len(A) > 0:
            # A集合的均方误差
            MSE_A = self.fit_and_compute_mse(X[:, A], y)
            MSE_min = float('inf')
            j_min = -1
            for j in A:
                # 去掉j特征列后的均方误差
                A_copy = A.copy()
                A_copy.remove(j)
                MSE_j = self.fit_and_compute_mse(X[:, A_copy], y)
                if MSE_j < MSE_min:
                    MSE_min, j_min = MSE_j, j
            # F检验
            if self.backward_f_test(MSE_A, MSE_min, m, confidence_interval=confidence_interval):
                A.remove(j_min)
                C.append(j_min)
            else:
                break
        self.A = A

    def train(self, X, y, type=StepwiseRegressionType.forward_selection,confidence_interval = 0.95):
        '''
        需要注明是向前还是向后回归
        '''
        if type == self.StepwiseRegressionType.forward_selection:
            self.forward_selection(X, y,confidence_interval=confidence_interval)
            self.w = self.fit(X[:, self.A], y)
        if type == self.StepwiseRegressionType.backward_selection:
            self.backward_selection(X, y,confidence_interval=confidence_interval)
            self.w = self.fit(X[:, self.A], y)

    def predict(self, X):
        return X[:, self.A].dot(self.w)
