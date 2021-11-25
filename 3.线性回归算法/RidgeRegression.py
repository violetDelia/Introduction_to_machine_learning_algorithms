import numpy as np
from LinearRegression import LinearRegression


class RidgeRegression(LinearRegression):
    def __init__(self, Lambda):
        self.Lambda = Lambda

    def train(self,X,y):
        m, n = X.shape
        r = m * np.diag(self.Lambda*np.ones(n))
        self.w = np.linalg.inv(X.T.dot(X) +r).dot(X.T).dot(y)