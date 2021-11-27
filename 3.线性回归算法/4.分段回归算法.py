import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from StagewiseRegression import StagewiseRegression
from StepwiseRegression import StepwiseRegression


def generate_sample(m):
    np.random.seed(int(time.time()))
    X = 2*(np.random.rand(m, 1)-0.5)
    y = X**5+X**2+X + np.random.normal(0, 0.1, (m, 1))
    return X, y


def show_results(model, X, y, degree):
    plt.scatter(X, y)
    line_x = np.linspace(X.min(), X.max(), 100).reshape(100, 1)
    line_x_ploy = PolynomialFeatures(degree=degree).fit_transform(line_x)
    line_y = model.predict(line_x_ploy)
    plt.scatter(line_x, line_y, color="red")
    plt.show()


def test_1():
    X, y = generate_sample(100)
    ploy_10 = PolynomialFeatures(degree=10)
    X_ploy = ploy_10.fit_transform(X)
    model = StagewiseRegression()
    model.train(X_ploy, y)
    print("\nw值: ", model.w)
    print("均方误差: ", model.MSE(y, model.predict(X_ploy)))
    print("R2: ", model.R2_score(y, model.predict(X_ploy)))
    show_results(model, X, y, 10)


def test_2():
    X = np.array(
        [[6, 34, 3], [44, 76, 28], [86, 44, 30], [26, 9, 25]])
    y = np.array([[42], [132], [84], [61]])
    '''
    逐步回归不适用这种情况情况
    '''
    model = StepwiseRegression()
    model.train(X, y)
    print(model.w)
    model_stage = StagewiseRegression()
    model_stage.train(X, y)
    y_predict = model_stage.predict(X)
    print(model_stage.w)
    plt.scatter(y, y_predict)
    plt.show()


if __name__ == "__main__":
    # 过于保守,导致避免过度拟合效果不足
    test_1()
    # test_2()
