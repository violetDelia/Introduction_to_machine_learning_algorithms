from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection
from Perceptron.Perceptron import Perceptron
import numpy as np

def get_data():
    X, y = make_blobs(n_samples=200, n_features=2,
                      cluster_std=0.6, random_state=0)
    data = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "lable": y})
    data["lable"] = data["lable"].replace({0: 1, 2:1,1:-1})
    return data


def show_data(data):
    sns.relplot(x=data["x"], y=data["y"], hue=data["lable"])
    plt.show()

def show_results(data, x_column, y_column, color_column, w, b):
    line_x = np.arange(data[x_column].agg(np.min), data[x_column].agg(np.max))
    line_y = (-b-w[0]*line_x)/w[1]
    sns.relplot(x=x_column, y=y_column, hue=color_column, data=data)
    plt.plot(line_x, line_y)
    plt.show()

def get_accuracy(predict_y, test_y):
    correct = 0
    for i in range(predict_y.size):
        if(predict_y[i] == test_y[i]):
            correct += 1
    return correct/predict_y.size

if __name__ == "__main__":
    data = get_data()
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(data[["x","y"]],data["lable"],test_size=0.3)
    model = Perceptron()
    model.train(train_x.values, train_y.values)
    show_results(data,"x","y","lable",model.w, model.b)
    predict_y = model.predict(test_x.values)
    print(model.get_accuracy(predict_y, test_y.values))

