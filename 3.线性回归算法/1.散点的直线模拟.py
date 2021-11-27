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


def show_scatter(X_train, y_train, y_train_predict, X_test, y_test, y_predict, column=1):
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(X_train[:, column], y_train)
    plt.scatter(X_train[:, column], y_train_predict, color='red')
    plt.subplot(212)
    plt.scatter(X_test[:, column], y_test)
    plt.scatter(X_test[:, column], y_predict, color='red')
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
    MES_test = model.MSE(y_test, y_predict)
    R2_test = model.R2_score(y_test, y_predict)
    print("测试集的均方误差和决定系数: ", MES_test, R2_test)
    y_train_line = model.predict(X_train)
    MES_train = model.MSE(y_train, y_train_line)
    R2_train = model.R2_score(y_train, y_train_line)
    print("训练集的均方误差和决定系数: ", MES_train, R2_train)
    show_scatter(X_train, y_train, y_train_line,
                 X_test, y_test, y_predict)
