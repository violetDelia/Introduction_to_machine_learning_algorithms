from enum import Enum
from LinearRegressionBase import LinearRegressionBase
from EvaluationModel import EvaluationModel
from DateVisualization import DateVisualization


class LinearRegression(LinearRegressionBase, EvaluationModel, DateVisualization):
    '''
    线性回归模型
    '''
    class SoulutionType(Enum):
        '''
        求解类型：

        normal 正规方程求解
        '''
        normal = 1
        '''
        正规方程求解
        '''
    

    class ProcessingType(Enum):
        '''
        特征处理的类型

        normal: 常规处理
        multinomial: 多项式处理
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

    def fit(self, X_train, y_train, type=SoulutionType.normal,processingType = ProcessingType.normal):
        '''
        训练模型

        参数：
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            type: 求解类型
        '''


        if type == self.SoulutionType.normal:
            self._fit_normal(X_train, y_train)
