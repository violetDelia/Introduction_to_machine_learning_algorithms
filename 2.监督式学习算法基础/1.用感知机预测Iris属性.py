import numpy as np
from Perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection


def load_data(remove_column):
    filepath = "2.监督式学习算法基础\Iris.csv"
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
def show_results(data,x_column,y_column,color_column,w,b):
    data["function_value"] = data[x_column]*w[0]+data[y_column]*w[1]+b
    print(data[["function_value","lable"]])

    


if __name__ == "__main__":
    remove_column = ["Id"]
    data = load_data(remove_column)
    set_lable(data)
    #show_scatter(data,data.columns[0],data.columns[1],data["lable"])
    train_x, test_x, train_y, test_y = get_train_and_test_data(
        data[[data.columns[0], data.columns[1]]], data["lable"], 0.3, 5)
    model = Perceptron()
    print(model.__dict__)
    model.train(train_x.values, train_y.values)
    print(model)
    show_results(data,data.columns[0],data.columns[1],data["lable"],model.w,model.b)
