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
        normal = 0
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

    class RegressionType(Enum):
        '''
        回归类型

        LinearRegression: 普通线性回归
        RigidRegression: 岭回归
        '''
        LinearRegression = 0
        '''
        普通线性回归
        '''
        RigidRegression = 1
        '''
        岭回归
        '''

    def __init__(self,):
        pass

    def _preprocessing(self, X_train, y_train, processingType, weights):
        '''
        数据集预处理

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            soulutionType: 求解类型
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
        y_processed = y_train
        return X_processed, y_processed

    def train(self, X_train, y_train, RegressionType=RegressionType.LinearRegression,
              soulutionType=SoulutionType.normal, processingType=ProcessingType.multinomial,
              weights=None):
        '''
        训练模型

        参数：
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            soulutionType: 求解类型
            processingType: 数据处理类型
            weights: 权重向量
        '''
        X_processed, y_processed = self._preprocessing(
            X_train, y_train, processingType, weights)
        if RegressionType == self.RegressionType.LinearRegression and soulutionType == self.SoulutionType.normal:
            self._fit_linear_normal(X_processed, y_processed)

    def predict(self, X, processingType=ProcessingType.multinomial, weights=None):
        '''
        用当前的w预测X对应的y

        参数:
            X: 预测用的特征集
            weights: 权重向量
        '''
        X_processed, y_processed = self._preprocessing(
            X, None, processingType, weights)
        return X_processed.dot(self.w)
