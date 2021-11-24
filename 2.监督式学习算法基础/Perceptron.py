import numpy as np
import pandas as pd


class Perceptron:
    '''学习率'''
    learning_rate = 1

    '''设置学习率'''
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def train(self, X, Y):
        m, n = X.shape
        '''初始w'''
        w = np.zeros((n, 1))
        '''初始b'''
        b = 0
        '''寻找到的标记'''
        is_find = False
        while not is_find:
            is_find = True
            '''
            遍历每个训练点，如果全都满足规则，跳出循环（可以设置为一定比例满足或者损失函数小于某个值）
            '''
            for i in range(m):
                x = X[i].reshape(1, -1)
                judge_value = Y[i] * (x.dot(w)+b)
                if judge_value <= 0:
                    '''
                    迭代w，b
                    '''
                    w = w + Y[i] * x.T*self.learning_rate
                    b = b+Y[i]*self.learning_rate
                    is_find = False
        self.w = w
        self.b = b

    def predict(self, X):
        return np.sign(X.dot(self.w)+self.b)
