import numpy as np
from LassoRegression import LassoRegression
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import PolynomialFeatures
from GDLinearRegression import GDLinearRegression
from LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression

def generate_sample(m):
    np.random.seed(int(time.time()))
    X = 2*(np.random.rand(m, 1)-0.5)
    y = X**4+X**2 + np.random.normal(0, 0.3, (m, 1))
    return X, y


if __name__ == "__main__":
    X, y = generate_sample(100)
    X_ploy = PolynomialFeatures(degree=10).fit_transform(X)
    model_1 = LinearRegression()
    model_2 = GDLinearRegression(using_epsilon = True)
    model_3 = LassoRegression(0.01)
    model_4 = LassoRegression(0.05)
    model_5 = LassoRegression(0.1)
    model_6 = LassoRegression(0.5)
    model_1.train(X_ploy, y)
    model_2.train(X_ploy, y)
    model_3.train(X_ploy, y)
    model_4.train(X_ploy, y)
    model_5.train(X_ploy, y)
    model_6.train(X_ploy, y)
    line_x = np.linspace(X.min(), X.max(), 50).reshape(-1, 1)
    line_x_ploy = PolynomialFeatures(degree=10).fit_transform(line_x)
    line_y_1 = model_1.predict(line_x_ploy)
    line_y_2 = model_2.predict(line_x_ploy)
    line_y_3 = model_3.predict(line_x_ploy)
    line_y_4 = model_4.predict(line_x_ploy)
    line_y_5 = model_5.predict(line_x_ploy)
    line_y_6 = model_6.predict(line_x_ploy)
    R2 = []
    R2.append(model_1.R2_score(y,model_1.predict(X_ploy)))
    R2.append(model_2.R2_score(y,model_2.predict(X_ploy)))
    R2.append(model_3.R2_score(y,model_3.predict(X_ploy)))
    R2.append(model_4.R2_score(y,model_4.predict(X_ploy)))
    R2.append(model_5.R2_score(y,model_5.predict(X_ploy)))
    R2.append(model_6.R2_score(y,model_6.predict(X_ploy)))
    print (R2)

    plt.scatter(X, y)
    plt.plot(line_x, line_y_1, color='red')
    plt.plot(line_x, line_y_2, color='green')
    plt.plot(line_x, line_y_3, color='black')
    plt.plot(line_x,line_y_4, color='black', linestyle='-.')
    plt.plot(line_x,line_y_5, color='black',linestyle=':')
    plt.plot(line_x,line_y_6, color='black', linestyle=' ')
    plt.show()
