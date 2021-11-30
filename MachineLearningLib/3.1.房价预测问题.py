import numpy as np
from LinearRegression.LinearRegression import LinearRegression
from Preprocessing.Preprocessing import Preprocessing
from sklearn.datasets import fetch_california_housing
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ModelEvaluation.ModelEvaluation import ModelEvaluation


def get_data():
    X = fetch_california_housing().data
    y = fetch_california_housing().target.reshape(-1, 1)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
   
    X_train, X_test, y_train, y_test = get_data()
    model = LinearRegression()
    model.train(X_train, y_train, processingType=model.ProcessingType.normal)
    y_predict = model.predict(
        X_test, processingType=model.ProcessingType.normal)
    print("MES = ", model.MSE(y_test, y_predict))
    print(" R2 = ", model.R2_score(y_test, y_predict))
    plt.figure()
    plt.subplot(121)
    model.plot_true_scatter_and_predict_scatter(X_train, y_train, model.predict(
        X_train, processingType=model.ProcessingType.normal), 2)
    plt.subplot(122)
    model.plot_true_scatter_and_predict_scatter(X_test, y_test, y_predict, 2)
    plt.show()
