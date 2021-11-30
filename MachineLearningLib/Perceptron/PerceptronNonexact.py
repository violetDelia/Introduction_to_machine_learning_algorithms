import numpy as np
from Perceptron.PerceptronMixin import PerceptronMixin


class PerceptronNonexact(PerceptronMixin):
    '''
    非精确求解的感知机

    参数：
        nonexact_accuracy_rate : 非精确求解时需要满足的合格率
            如果采用非精确求解,那么每迭代NonExactSolution_step次后会检验一次合格率,如果预测满足该合格率,那么认为该解是可行解

        nonexact_steps: 非精确求解检验合格率的迭代参数
    '''

    nonexact_accuracy_rate = None
    nonexact_steps = None

    def _set_nonexact_accuracy_rate(self, nonexact_accuracy_rate):
        self.nonexact_accuracy_rate = 0.95
        if isinstance(nonexact_accuracy_rate, float) and nonexact_accuracy_rate > 0 and nonexact_accuracy_rate < 1:
            self.nonexact_accuracy_rate = nonexact_accuracy_rate

    def _set_nonexact_steps(self, nonexact_steps):
        self.nonexact_steps = 50
        if isinstance(nonexact_steps, int) and nonexact_steps > 0:
            self.nonexact_steps = nonexact_steps
        if isinstance(nonexact_steps, float) and nonexact_steps > 1:
            self.nonexact_steps = int(nonexact_steps)

    def _find_nonexact_solution(self, X_check, y_check, w_check, b_check):
        '''
        是否找到了非精确解。

        参数:
            X_check: 检测用的样本
            y_check: 检测用的标签
            w_check: 检测用的w值
            b_check: 检测用的b值

        '''
        current_predict = self._predict(X_check, w_check, b_check)
        current_accuracy = self.get_accuracy(y_check, current_predict)
        return current_accuracy > self.nonexact_accuracy_rate

    def _need_nonexact_check(self, steps):
        '''
        是否需要非精确合格率检验。

        参数：
            steps: 计算迭代的次数
        '''
        return steps % self.nonexact_steps == 0

    def _fit_nonexact(self, X_train, y_train):
        '''
        寻找符合假设的超平面并更新类属性w,b

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
        '''

        m, n = X_train.shape
        w_current, b_current = self._get_initial_points(n)
        is_find = False
        steps = 0
        while self._need_continue_find_solution(is_find, steps):
            is_find = True
            if self._need_nonexact_check(steps):
                if self._find_nonexact_solution(X_train, y_train, w_current, b_current):
                    break
            for i in range(m):
                x = X_train[i].reshape(1, -1)
                if self._sample_is_qualify(x, i, y_train, w_current, b_current):
                    w_current, b_current = self._find_w_and_b(
                        x, i, y_train, w_current, b_current)
                    is_find = False
            steps += 1
        print("采用非精确求解,非精确搜索合格率为{0}".format(
            self.nonexact_accuracy_rate))
        self._show_solution_base(is_find, steps, w_current, b_current)

    def _predict(self, X_test, w, b):
        '''
        用w,b来预测标签值

        参数:
            X_test: 预测用的数据集
            w: 预测用的w值
            b: 预测用的b值

        返回值:
            y_predict: 预测的标签值
        '''
        y_predict = np.sign(X_test.dot(w)+b)
        return y_predict
