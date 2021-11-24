import numpy as np

class Perceptron:
    def train(self,X,Y):
        m,n = X.shape
        print(X.shape)
        w = np.zeros((n,1))
        b = 0
        is_finished = False
        while not is_finished:
            is_finished = True
            for i in range(m):
                x = X[i].reshape(1,-1)
                if Y[i] *(x.dot(w)+b <= 0):
                    w = w +Y[i] *x.T
                    b = b+Y[i]
                    is_finished = False
        self.w = w
        self.b = b

    def predict(self,X):
        return np.sign(X.dot(self.w)+self.b)

