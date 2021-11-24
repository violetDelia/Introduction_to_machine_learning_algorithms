import numpy as np


class Perceptron:
    def train(self, X, Y):
        m, n = X.shape
        print(X.shape)
        w = np.zeros((n, 1))
        b = 0
        is_find = False
        while not is_find:
            is_find = True
            for i in range(m):
                x = X[i].reshape(1, -1)
                if Y[i] * (x.dot(w)+b <= 0):
                    w = w + Y[i] * x.T
                    b = b+Y[i]
                    is_find = False
        self.w = w
        self.b = b

    def predict(self, X):
        return np.sign(X.dot(self.w)+self.b)
