import numpy as np
from GDLinearRegression import GDLinearRegression


class LassoRegression(GDLinearRegression):

    def __init__(self,Lambda=0.01, w=None, learning_rate=0.5, max_steps=2000):
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.w = w
        self.Lambda = Lambda


    def fit(self, X, y):
        m, n = X.shape
        if self.w == None:
            w = np.ones((n, 1))
        else:
            w = self.w
        sum_w = w
        for i in range(self.max_steps):
            v = 2*X.T.dot(X.dot(w)-y)/m+self.Lambda*np.sign(w)
            w = w-self.learning_rate*v
            sum_w += w
        self.w = sum_w/self.max_steps
