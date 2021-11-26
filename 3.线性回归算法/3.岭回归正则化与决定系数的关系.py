import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures
from RidgeRegression import RidgeRegression
import matplotlib.pyplot as plt

'''
R2最接近的lambda一般是最优的lambda，可以在过度拟合和拟合不足之间取得良好的平衡。
'''
def generate_sample(m):
    np.random.seed()
    X = 2*np.random.rand(m,1)
    y = X**2 - 1 + np.random.normal(0,0.1,(m,1))
    return X,y

if __name__ == "__main__":
    ploy_10 = PolynomialFeatures(degree = 10)
    X_train,y_train = generate_sample(30)
    X_train_ploy = ploy_10.fit_transform(X_train)
    X_test,y_test = generate_sample(100)
    X_test_ploy = ploy_10.fit_transform(X_test)

    Lambdas,train_R2s,test_R2s = [],[],[]
    for i in range(200):
        Lambda = 0.01*i
        Lambdas.append(Lambda)
        model = RidgeRegression(Lambda)
        model.train(X_train_ploy,y_train)
        y_train_predict = model.predict(X_train_ploy)
        y_test_predict = model.predict(X_test_ploy)
        train_R2s .append(model.R2_score(y_train,y_train_predict))
        test_R2s .append(model.R2_score(y_test, y_test_predict))

    plt.figure(1)
    plt.plot(Lambdas,train_R2s)
    plt.plot(Lambdas,test_R2s,color = 'red')
    plt.show()
    plt.figure(2)
    distances = []
    for train_R2,test_R2 in zip(train_R2s,test_R2s):
        distances.append(train_R2-test_R2)
    plt.plot(distances)
    plt.show()