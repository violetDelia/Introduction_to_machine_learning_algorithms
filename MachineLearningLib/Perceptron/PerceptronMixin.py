import numpy as np


class PerceptronMixin:
    '''
    最基础的感知机

    属性:
        learning_rate: 学习率
                迭代搜索符合假设的超平面时的步长

        w: 各个特征的权重
            是一个(n,1)的数组.

        b: 偏移量

        start_w: 搜索w的初始点
            如果设置了,那么会从start_w开始搜索解的w,否则,会从零向量开始搜索.

        start_b: 搜索b的初始点
            如果设置了,那么会从start_b开始搜索解的b,否则,会从0开始搜索.

        max_steps: 迭代的最大次数
            如果迭代超过max_steps还没有找到解的话,退出搜索,不更新w,b值.
    '''
    learning_rate = None
    w_start = None
    w = None
    b_start = None
    b = None
    max_steps = None

    def _set_learning_rate(self, learning_rate):
        self.learning_rate = 1.0
        if learning_rate > 0:
            self.learning_rate = learning_rate

    def _set_w_start(self, w_start):
        self.w_start = None
        if isinstance(w_start, list):
            self.w_start = np.array(w_start).reshape(-1, 1)
        if isinstance(w_start, np.ndarray):
            self.w_start = w_start.reshape(-1, 1)

    def _set_b_start(self, b_start):
        self.b_start = 0
        if isinstance(b_start, int):
            self.b_start = float(b_start)
        if isinstance(b_start, float):
            self.b_start = b_start

    def _set_max_steps(self, max_steps):
        self.max_steps = 3000
        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        if isinstance(max_steps, float) and max_steps > 1:
            self.max_steps = int(max_steps)

    def _get_initial_points(self, n):
        '''
        获取初始点.

        参数:
            n: 特征个数

        返回:
            w_start: 搜索w的初始点
            b_start: 搜索b的初始点
        '''
        if self.w_start is not None and self.w_start.shape == (n, 1):
            w_start = self.w_start
        else:
            w_start = np.zeros((n, 1))
        b_start = self.b_start
        return w_start, b_start

    def _show_solution_base(self, is_find, steps, w, b):
        '''
        输出最基本的计算结果

        参数：
            is_find: 是否找到解的标记
            steps: 计算迭代的次数
            w: 迭代结束时的w
            b: 迭代结束时的b
        '''
        print("感知机迭代次数: ", steps)
        if is_find:
            self.w = w
            self.b = b
            print("    寻找到w: ", w.reshape(1, -1))
            print("    寻找到b: ", b)
        else:
            print("最大迭代数为{0},未找到解.".format(self.max_steps))

    def _need_continue_find_solution(self, is_find, steps):
        '''
        是否继续迭代

        参数：
            is_find: 是否找到解的标记
            steps: 计算迭代的次数
        '''
        return not is_find and steps < self.max_steps

    def _sample_is_qualify(self, x, i, y_train, w_current, b_current):
        '''
        当前的w_current，b_current能否让该样本合格

        参数：
            x: 选取样本的特征值
            i: 样本的位置
            y_train: 训练的标签集
            w_current: 当前计算的w
            b_current: 当前计算的b

        '''
        judge_value = y_train[i] * (x.dot(w_current)+b_current)
        return judge_value <= 0

    def _find_w_and_b(self, x, i, y_train, w_current, b_current):
        '''
        寻找能使该样本合格的w和b

        参数：
            x: 选取样本的特征值
            i: 样本的位置
            y_train: 训练的标签集
            w_current: 当前计算的w
            b_current: 当前计算的b

        返回值：
            w：使该样本合格的w
            b：使该样本合格的b
        '''
        w = w_current + y_train[i] * x.T*self.learning_rate
        b = b_current + y_train[i]*self.learning_rate
        return w, b

    def _fit_normal(self, X_train, y_train):
        '''
        基本感知机的搜索算法

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
            for i in range(m):
                x = X_train[i].reshape(1, -1)
                if self._sample_is_qualify(x, i, y_train, w_current, b_current):
                    w_current, b_current = self._find_w_and_b(
                        x, i, y_train, w_current, b_current)
                    is_find = False
            steps += 1
        self._show_solution_base(is_find, steps, w_current, b_current)
