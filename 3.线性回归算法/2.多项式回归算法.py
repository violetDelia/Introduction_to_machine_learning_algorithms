import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt

def generate_sample(m):
    np.random.seed(int(time.time()))
    X = 2*np.random.rand(m,1)
    y = X**2 - 1 + np.random.normal(0,0.1,(m,1))
    return X,y

if __name__ == "__main__":
    X,y = generate_sample(100)
    X_ploy = PolynomialFeatures(degree= 2).fit_transform(X)
    model = LinearRegression()
    model.train(X_ploy,y)
    y_predict = model.predict(X_ploy)
    plt.scatter(X,y)
    plt.scatter(X,y_predict,color='red')
    plt.show()
