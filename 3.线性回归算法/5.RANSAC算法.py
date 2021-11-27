import numpy as np
import time
import matplotlib.pyplot as plt
from RANSACLinearRegression import RANSACLinearRegression
from sklearn.linear_model import RANSACRegressor


def generate_sample(m,k):
    np.random.seed(int(time.time()))
    X_normal = 2*(np.random.rand(m, 1)-0.5)
    y_normal = 2*X_normal+ np.random.normal(0,0.1,(m, 1))
    X_outlier = 2*(np.random.rand(k, 1)-0.5)
    y_outlier = X_outlier+ np.random.normal(3,0.1,(k, 1))
    X = np.concatenate((X_normal,X_outlier),axis=0)
    y = np.concatenate((y_normal,y_outlier),axis=0)
    return X,y

if __name__ == "__main__":
    X,y = generate_sample(100,10)
    model = RANSACRegressor()
    model.fit(X,y)
    model_My = RANSACLinearRegression(1)
    model_My.train(X,y)
    line_x = np.linspace(X.min(),X.max(),50).reshape(-1, 1)
    line_y = model.predict(line_x)
    line_my_y = model_My.predict(line_x)
    plt.scatter(X,y)
    plt.plot(line_x,line_y, color="red")
    plt.plot(line_x,line_my_y, color="green")
    plt.show()