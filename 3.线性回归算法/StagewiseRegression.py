import numpy as np
from LinearRegression import LinearRegression


class StagewiseRegression(LinearRegression):
    def __init__(self, learning_rate=0.01, max_steps=2000):
        self.learning_rate = learning_rate
        self.max_steps = max_steps

    def train(self, X, y):
        m, n = X.shape
        w = np.zeros(n)
        # 计算每列的二范数
        norms = np.linalg.norm(X, 2, axis=0).reshape(-1, 1)
        step = 0
        # 误差向量
        distances = y
        while step < self.max_steps:
            # 相关系数
            corr = X.T.dot(distances)/norms
            # 相关性最高的那组特征
            j_max = np.argmax(abs(corr))
            if w[j_max] == 0:
                w[j_max] = self.learning_rate
            else:
                w[j_max] = w[j_max] + self.learning_rate * np.sign(corr[j_max])
            distances = distances - self.learning_rate * \
                X[:, j_max].reshape(-1, 1)
            step += 1
        self.w = w
        return w

    def predict(self, X):
        return X.dot(self.w)
