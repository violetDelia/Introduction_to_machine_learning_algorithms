import numpy as np
class LinearRegressionBase:

    def _regularization(self,X,y):
        '''
        用正则化方法求得w

        参数：
            X：处理好的训练特征集
            y：处理好的训练结果集
        '''
        

    def _fit_regularization(self,X_train, y_train):
        '''
        使用正则化方程直接求得最优解

        参数:
            X_train: 训练特征集
            y_train: 训练结果集
        '''
        self.w = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

    
