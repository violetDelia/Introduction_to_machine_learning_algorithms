import numpy as np
from LinearRegression import LinearRegression
import time
def generate_sample(m):
    X = 2*(np.random.rand(m,1)-0.5)
    y = X + np.random.normal(0,0.3,(m,1))
    return X,y

def process_features(X):
    m,n = X.shape
    X = np.c_[np.ones((m,1)),X]
    return X

if __name__ == "__main__":
    generate_sample(100)
    np.random.seed(int(time.time()))
    X_train ,y_train = generate_sample(5)
    X_train = process_features(X_train)
    X_test,y_test = generate_sample(100)
    X_test = process_features(X_test)




