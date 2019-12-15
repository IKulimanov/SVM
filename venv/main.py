from pandas import read_csv, DataFrame, Series
#from pydotplus import graph_from_dot_data
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from matplotlib import mlab

train_dataset = read_csv('train.csv')

train_dataset.info()

dep_var = train_dataset["Survived"] #зависимая переменная
train_dataset.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#test_dataset.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#нормализация данных
def harmonize_data(titanic):
    #отсутствующим полям возраста присваивается медианное значение
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()
    #пол преобразуется в числовой формат
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    #пустое место отплытия заполняется наиболее популярным S
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    # место отплытия преобразуется в числовой формат
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    # отсутствующим полям суммы отплаты за плавание присваивается медианное значение
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    return titanic
train_harm = harmonize_data(train_dataset)
train_harm.info()
clf_svm = SVC()

npX = np.array(train_harm).copy()
npy = np.array(dep_var).copy()


parameters_svm = {'C':[0.9,0.01],'kernel':['rbf','linear'], 'gamma':[0,0.1,'auto'], 'probability':[True,False],
                  'random_state':[0,7,16],'decision_function_shape':['ovo','ovr'],'degree':[3,4,10]}
def grid_fun(model, parameters):
    grid = GridSearchCV(estimator = model, param_grid = parameters, cv = 10,
                        scoring = 'accuracy')
    grid.fit(npX, npy)
    return grid.best_score_, grid.best_estimator_.get_params()

best_score_svm, best_params_svm = grid_fun(clf_svm, parameters_svm)
print(best_score_svm)

