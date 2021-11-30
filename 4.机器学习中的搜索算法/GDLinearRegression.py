import numpy as np
from LinearRegression import LinearRegression


class GDLinearRegression(LinearRegression):

    def __init__(self, w=None, learning_rate=0.5, max_steps=2000, using_epsilon=False, epsilon=1e-3):
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.w = w
        self.using_epsilon = using_epsilon
        self.epsilon = epsilon

    def fit(self, X, y):
        m, n = X.shape
        if self.w == None:
            w = np.ones((n, 1))
        else:
            w = self.w
        for i in range(self.max_steps):
            e = X.dot(w)-y
            g = 2*X.T.dot(e)/m
            if((self.using_epsilon == True) & (g[:,:].max() < self.epsilon)):
                print ("使用梯度下降法第{0}次迭代结束,eps为: {1}".format(i,self.epsilon))
                break
            w = w-self.learning_rate*g
        self.w = w

    def train(self, X, y):
        self.fit(X, y)
