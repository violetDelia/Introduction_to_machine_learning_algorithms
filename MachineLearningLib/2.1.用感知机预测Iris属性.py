import numpy as np
from Perceptron.Perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection


def load_data(remove_column):
    filepath = "MachineLearningLib\datasets\Iris\Iris.csv"
    data = pd.read_csv(filepath)
    '''
    去掉没用的列
    '''
    data.drop(remove_column, axis=1, inplace=True)
    return data


def show_scatter(data, x_column, y_column, color_column=None, size_column=None, sizes=None):
    sns.relplot(x=x_column, y=y_column, hue=color_column,
                data=data, size=size_column, sizes=sizes)

    plt.show()


def get_train_and_test_data(data, lable, test_size, random_state):
    return sklearn.model_selection.train_test_split(data, lable, test_size=test_size, random_state=random_state)


def set_lable(data):
    lable_group = data.groupby([data.columns[4]])
    lable_list = []
    for name, info in lable_group:
        lable_list.append(name)
    data["lable"] = data[data.columns[4]].replace(
        {lable_list[0]: -1, lable_list[1]: 1, lable_list[2]: 1})


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
    remove_column = ["Id"]
    data = load_data(remove_column)
    set_lable(data)
    # show_scatter(data,data.columns[0],data.columns[1],data["lable"])
    train_x, test_x, train_y, test_y = get_train_and_test_data(
        data[[data.columns[0], data.columns[1]]], data["lable"], 0.3, 5)
    model = Perceptron(w_start=[55.9, -80.9], b_start=50)
    model.train(train_x.values, train_y.values,
                model.SoulutionType.normal_nonexact)
    show_results(
        data, data.columns[0], data.columns[1], data["lable"], model.w, model.b)
    predict_y = model.predict(test_x.values)
    print(predict_y.shape)
    print(type(test_y.values))
    print(model.get_accuracy(predict_y, test_y.values))
    # 记录信息
    '''
    train_x.to_csv("2.监督式学习算法基础\Iris_compute_info\\train_x.csv")
    train_y.to_csv("2.监督式学习算法基础\Iris_compute_info\\train_y.csv")
    test_x.to_csv("2.监督式学习算法基础\Iris_compute_info\\test_x.csv")
    test_y.to_csv("2.监督式学习算法基础\Iris_compute_info\\test_y.csv")
    for i in range(len(model.compute_info)):
        filename = "2.监督式学习算法基础\Iris_compute_info\第{0}次迭代.csv".format(i)
        model.compute_info[i].to_csv(filename
        )
    test_x["lable"] = test_y
    test_x["predict"] = pd.DataFrame(pd.Series(predict_y.reshape(1, -1)[0],index=test_x.index))
    test_x.to_csv("2.监督式学习算法基础\Iris_compute_info\预测结果.csv")
    '''
