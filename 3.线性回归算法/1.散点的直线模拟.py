import numpy as np
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
import time


def generate_sample(m):
    X = 2*(np.random.rand(m, 1)-0.5)
    y = X + np.random.normal(0, 0.3, (m, 1))
    return X, y


def process_features(X):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X


def show_scatter(x_train_data, y_train_data, y_train_line,x_test_data, y_test_data, y_predict):
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(x_train_data[:, 1], y_train_data)
    plt.plot(x_train_data[:, 1],y_train_line, color='red')
    plt.subplot(212)
    plt.scatter(x_test_data[:, 1], y_test_data)
    plt.plot(x_test_data[:, 1], y_predict, color='red')
    plt.show()


if __name__ == "__main__":
    generate_sample(100)
    np.random.seed(int(time.time()))
    X_train, y_train = generate_sample(50)
    X_train = process_features(X_train)
    X_test, y_test = generate_sample(50)
    X_test = process_features(X_test)
    model = LinearRegression()
    model.train(X_train, y_train)
    y_predict = model.predict(X_test)
    MES_test = model.mean_squared_error(y_test, y_predict)
    R2_test = model.R2_score(y_test, y_predict)
    print("测试集的均方误差和决定系数: ", MES_test, R2_test)
    y_train_line = model.predict(X_train)
    MES_train = model.mean_squared_error(y_train, y_train_line)
    R2_train = model.R2_score(y_train, y_train_line)
    print("训练集的均方误差和决定系数: ", MES_train, R2_train)
    show_scatter(X_train, y_train, y_train_line,
                 X_test, y_test, y_predict)
