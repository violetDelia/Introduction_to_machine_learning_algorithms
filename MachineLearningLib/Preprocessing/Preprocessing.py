import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


class Preprocessing:
    '''
    数据处理基类
    '''

    def _add_onevector_to_features(self, X):
        '''
        给特征值加一组全是1的向量

        参数：
            X: 要处理的特征集
        '''
        X_standaardized = StandardScaler().fit_transform(X)
        m, n = X.shape
        X_addone = np.c_[np.ones((m, 1)), X_standaardized]
        return X_addone

    def _process_feature_to_multinomial_features(self, X, degree=2):
        '''
        特征组多项式处理

        参数：
            X: 要处理的特征集
            degree: 多项式次数
        '''
        X_processed = PolynomialFeatures(degree=degree).fit_transform(X)
        return X_processed

    def _process_weight_to_features(self, X, weights):
        '''
        给特征集加入权重

        参数:
            X: 要处理的特征集
            weights: 权重向量
        '''
        m, n = X.shape
        can_be_processed = False
        if isinstance(weights, list) and len(weights) == n:
            weights = np.array(weights).reshape(1, -1)
            can_be_processed = True
        if isinstance(weights, np.ndarray) and weights.size == n:
            weights = weights.reshape(1, -1)
            can_be_processed = True
        if can_be_processed:
            X = X*weights
        return X
