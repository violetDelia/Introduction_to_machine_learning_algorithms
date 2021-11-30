from enum import Enum
import numpy as np
from LinearRegression.LinearRegressionMixin import LinearRegressionMixin
from ModelEvaluation.ModelEvaluation import ModelEvaluation
from DataVisualization.DateVisualization import DataVisualization
from Preprocessing.Preprocessing import Preprocessing


class LinearRegression(LinearRegressionMixin, DataVisualization, Preprocessing, ModelEvaluation):
    '''
    线性回归模型
    '''
    class SoulutionType(Enum):
        '''
        求解类型：

        normal: 正规方程求解
        '''
        normal = 1
        '''
        正规方程求解
        '''

    class ProcessingType(Enum):
        '''
        特征处理的类型

        not_process: 不处理
        normal: 常规处理
        multinomial: 多项式处理
        '''
        not_process = 0
        '''
        不处理
        '''
        normal = 1
        '''
        常规处理
        '''
        multinomial = 2
        '''
        多项式处理
        '''

    def __init__(self,):
        pass

    def train(self, X_train, y_train, type=SoulutionType.normal, processingType=ProcessingType.multinomial,
              weights=None):
        '''
        训练模型

        参数：
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            type: 求解类型
            processingType: 数据处理类型
            weights: 权重向量
        '''

        if weights is not None:
            X_train = self._process_weight_to_features(X_train, weights)
        if processingType == self.ProcessingType.not_process:
            X_processed = X_train
        if processingType == self.ProcessingType.normal:
            X_processed = self._add_onevector_to_features(X_train)
        if processingType == self.ProcessingType.multinomial:
            X_processed = self._process_feature_to_multinomial_features(
                X_train)
        if type == self.SoulutionType.normal:
            self._fit_normal(X_processed, y_train)

    def predict(self, X, processingType=ProcessingType.multinomial, weights=None):
        '''
        用当前的w预测X对应的y

        参数:
            X: 预测用的特征集
            weights: 权重向量
        '''
        if weights is not None:
            X = self._process_weight_to_features(X, weights)
        if processingType == self.ProcessingType.not_process:
            X_processed = X
        if processingType == self.ProcessingType.normal:
            X_processed = self._add_onevector_to_features(X)
        if processingType == self.ProcessingType.multinomial:
            X_processed = self._process_feature_to_multinomial_features(
                X)
        return X_processed.dot(self.w)
