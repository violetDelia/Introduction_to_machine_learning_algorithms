import numpy as np
from LinearRegression import LinearRegression
from sklearn.datasets import fetch_california_housing
import sklearn.model_selection 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def process_features(X):
    #标准化
    X = StandardScaler().fit_transform(X)
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    return X

def get_data():
    X = fetch_california_housing().data
    y = fetch_california_housing().target.reshape(-1,1)
    X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.2,random_state = 0)
    X_train = process_features(X_train)
    X_test = process_features(X_test)
    return X_train,X_test,y_train,y_test
def show_scatter(X_train, y_train, y_train_predict,X_test, y_test, y_predict,column = 1):
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(X_train[:, column], y_train)
    plt.scatter(X_train[:, column],y_train_predict, color='red')
    plt.subplot(212)
    plt.scatter(X_test[:, column], y_test)
    plt.scatter(X_test[:, column], y_predict, color='red')
    plt.show()

if __name__ == "__main__":
   X_train,X_test,y_train,y_test= get_data()
   model = LinearRegression()
   model.train(X_train,y_train)
   y_predict = model.predict(X_test)
   MES = model.MSE(y_test, y_predict)
   R2 = model.R2_score(y_test, y_predict)
   print(MES, R2)
   show_scatter(X_train, y_train,model.predict(X_train),X_test, y_test, y_predict)
   

