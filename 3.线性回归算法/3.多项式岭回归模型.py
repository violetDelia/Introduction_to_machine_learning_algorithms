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

def show_results(model,X,y,degree):
    plt.scatter(X,y)
    line_x = np.linspace(X.min(),X.max(),100).reshape(100,1)
    line_x_ploy = PolynomialFeatures(degree=degree).fit_transform(line_x)
    line_y = model.predict(line_x_ploy)
    plt.plot(line_x, line_y,color="red")
    plt.show()


if __name__ == "__main__":
    X,y = generate_sample(40)
    X_ploy = PolynomialFeatures(degree=10).fit_transform(X)
    model = RidgeRegression()
    model.train(X_ploy,y)
    print(model.MSE(y,model.predict(X_ploy)))
    print(model.R2_score(y,model.predict(X_ploy)))
    show_results(model,X,y,10)
    
