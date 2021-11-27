import numpy as np
from LinearRegression import LinearRegression


class WeightedLinearRegression(LinearRegression):

    def process_weights(self, X, v):
        '''
        v 是关于权重的(1,n)矩阵
        '''
        m, n = X.shape
        V = np.zeros((n, n))
        for i in range(n):
            V[i, :] = v
        Xv = X*v
        return Xv

    def fit_weights(self, X, y, v):
        '''
        v 是关于权重的(1,n)矩阵
        '''
        Xv = self.process_weights(X, v)
        self.fit(Xv, y)

    def train(self, X, y, v):
        self.fit_weights(X, y, v)
