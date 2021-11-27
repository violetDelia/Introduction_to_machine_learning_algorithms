import numpy as np
from LinearRegression import LinearRegression
from RidgeRegression import RidgeRegression
from StepwiseRegression import StepwiseRegression
from StagewiseRegression import StagewiseRegression
from sklearn.datasets import load_diabetes
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def get_data(test_size=0.3, random_state=0):
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return X_train, X_test, y_train, y_test


def get_accuracy(y_predict, y_test):
    correct = 0
    for i in range(y_predict.size):
        if(abs(y_predict[i] - y_test[i]) <= 20):
            correct += 1
    return correct/y_predict.size


def UsingLinearRegression(X_train, X_test, y_train, y_test):

    def process_features(X):
        # 标准化
        X = StandardScaler().fit_transform(X)
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]
        return X

    X_train = process_features(X_train)
    X_test = process_features(X_test)
    model = LinearRegression()
    model.train(X_train, y_train)
    y_predict = model.predict(X_test)
    print(" MES = ", model.MSE(y_test, y_predict))
    print(" R2 = ", model.R2_score(y_test, y_predict))
    print(" Accuracy = ", get_accuracy(y_predict, y_test))


def UsingRidgeRegression(X_train, X_test, y_train, y_test, Lambda=0.01, poly_degree=2):
    X_train_ploy = PolynomialFeatures(
        degree=poly_degree).fit_transform(X_train)
    X_test_ploy = PolynomialFeatures(degree=poly_degree).fit_transform(X_test)
    model = RidgeRegression(Lambda)
    model.train(X_train_ploy, y_train)
    print("w值: ", model.w)
    y_predict = model.predict(X_test_ploy)
    print("均方误差: ", model.MSE(y_test, y_predict))
    print("R2: ", model.R2_score(y_test, y_predict))
    print(" Accuracy = ", get_accuracy(y_predict, y_test))


def UsingStepRegression(X_train, X_test, y_train, y_test, confidence_interval=0.95,  type=StepwiseRegression.StepwiseRegressionType.forward_selection, poly_degree=2,):

    X_train_ploy = PolynomialFeatures(
        degree=poly_degree).fit_transform(X_train)
    X_test_ploy = PolynomialFeatures(degree=poly_degree).fit_transform(X_test)
    model = StepwiseRegression()
    model.train(X_train_ploy, y_train, type, confidence_interval)
    print("选取的特征: ", model.A, "\nw值: ", model.w)
    y_predict = model.predict(X_test_ploy)
    print("均方误差: ", model.MSE(y_test, y_predict))
    print("R2: ", model.R2_score(y_test, y_predict))
    print(" Accuracy = ", get_accuracy(y_predict, y_test))


def UsingStageRegression(X_train, X_test, y_train, y_test, learning_rate=0.1, poly_degree=2):
    X_train_ploy = PolynomialFeatures(
        degree=poly_degree).fit_transform(X_train)
    X_test_ploy = PolynomialFeatures(degree=poly_degree).fit_transform(X_test)
    model = StagewiseRegression(learning_rate, 500/learning_rate)
    model.train(X_train_ploy, y_train)
    print("\nw值: ", model.w)
    y_predict = model.predict(X_test_ploy)
    print("均方误差: ", model.MSE(y_test, y_predict))
    print("R2: ", model.R2_score(y_test, y_predict))
    print(" Accuracy = ", get_accuracy(y_predict, y_test))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    # 0.37
    UsingLinearRegression(X_train, X_test, y_train, y_test)
    # 0.40
    UsingRidgeRegression(X_train, X_test, y_train, y_test,0.001)
    # 0.31
    UsingStepRegression(X_train, X_test, y_train, y_test,0.8,StepwiseRegression.StepwiseRegressionType.forward_selection)
    #UsingStageRegression(X_train, X_test, y_train, y_test)
