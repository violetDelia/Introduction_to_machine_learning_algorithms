import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from RidgeRegression import RidgeRegression


def generate_sample(m):
    np.random.seed(int(time.time()))
    X = 2*(np.random.rand(m, 1)-0.5)
    y = X + np.random.normal(0, 0.3, (m, 1))
    return X, y


def show_results(model, X, y, degree):
    plt.scatter(X, y)
    line_x = np.linspace(X.min(), X.max(), 100).reshape(100, 1)
    line_x_ploy = PolynomialFeatures(degree=degree).fit_transform(line_x)
    line_y = model.predict(line_x_ploy)
    plt.plot(line_x, line_y, color="red")
    plt.show()


if __name__ == "__main__":
    X_train, y_train = generate_sample(40)
    y_test = y_train

    X_train_ploy = PolynomialFeatures(degree=10).fit_transform(X_train)
    model = RidgeRegression()
    model.train(X_train_ploy, y_train)
    print("\nw值: ", model.w)
    y_predict = model.predict(X_train_ploy)
    print("均方误差: ", model.MSE(y_test, y_predict))
    print("R2: ", model.R2_score(y_test, y_predict))
    show_results(model, X_train, y_train, 10)
