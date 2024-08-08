import json

from abstract import Structure
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, OrthogonalMatchingPursuit
from sklearn.linear_model import ARDRegression, BayesianRidge, HuberRegressor, QuantileRegressor, RANSACRegressor
from sklearn.linear_model import TheilSenRegressor, GammaRegressor, PoissonRegressor, TweedieRegressor, PassiveAggressiveRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from main import FrameWork
import threading
import multiprocessing
from sklearn.multioutput import MultiOutputRegressor
import os
import pickle
import collections
import predict
import time
from initial import Initial


class Pipeline:
    def __init__(self, fer):
        fer.train_test_split_model()
        fer.standartization_data()
        fer.create_grid_model()
        fer.fit_model()
        fer.pred_model()
        fer.error_model()
        self.return_data = fer.save_test()

    def return_1(self):
        return self.return_data

def linear_family(alias_algoritm, memory, initial):
    time_initial = time.time()
    list_store_information = []
    models = initial[alias_algoritm]
    degree_iteration = models['poly_bool']
    for number_of_y_column in range(models['number_of_y']):
        for degree in range(1, degree_iteration + 1):
            fer = FrameWork(initial['data_path'],
                            models['number_of_x'], number_of_y_column, models['model'], models['percent_train'], models['random_seed'],
                            param_grid=models['param_grid'], cv=5, scoring=models['scoring'], poly_bool=degree, alias_model=alias_algoritm)
            if models['poly_bool'] != 1:
                fer.polyniminal_model()
            temp_pipline = Pipeline(fer)
            list_store_information.append(temp_pipline.return_1())
    memory.put(list_store_information)
    print(f'{alias_algoritm} за {round((time.time() - time_initial), 2)} с')

if __name__ == '__main__':
    best_result_models_info = {}
    result_models_info = {}
    with open('PSEVEN_INFO.json', 'r') as file:
        pseven_info = json.load(file)
    number_of_x = pseven_info['x']
    number_of_y = pseven_info['y']
    percent_train = pseven_info['percent_train']
    random_seed = pseven_info['random_seed']
    poly_bool = pseven_info['degree']
    metric_for_best = pseven_info['metric']                          # по ней идет поиск 'лучшей' модели
    data_path = pseven_info['data_path']
    if metric_for_best == 'mae':
        scoring = 'neg_mean_absolute_error'
    elif metric_for_best == 'mse':
        scoring = 'neg_mean_squared_error'
    else:
        scoring = 'neg_root_mean_squared_error'

    initial = Initial(data_path).initial


    for key, value in initial.items():
        if key != 'data_path':
            value['poly_bool'] = poly_bool
            value['number_of_x'] = number_of_x
            value['number_of_y'] = number_of_y
            value['percent_train'] = percent_train
            value['random_seed'] = random_seed
            value['scoring'] = scoring

    list_methods = ['LinearRegression', 'Ridge', 'Lasso', 'Elastic', 'K_Neighbours', 'DecisionTree', 'SVR', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'KernelRidge',
                     'OrthogonalMatchingPursuit', 'ARDRegression', 'BayesianRidge', 'QuantileRegressor',
                    'TheilSenRegressor', 'GammaRegressor', 'TweedieRegressor', 'PassiveAggressiveRegressor', 'RadiusNeighborsRegressor',
                    'BaggingRegressor', 'ExtraTreesRegressor', 'HistGradientBoostingRegressor']
    """list_methods = ['LinearRegression', 'Ridge', 'Lasso', 'Elastic', 'K_Neighbours', 'DecisionTree']"""
    """list_methods = ['LinearRegression']"""
    list_memory = [multiprocessing.Queue() for i in range(len(list_methods))]
    list_processes = [multiprocessing.Process(target=linear_family, args=(list_methods[i], list_memory[i], initial)) for i in range(len(list_methods))]
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('graphics'):
        os.mkdir('graphics')
    [i.start() for i in list_processes]
    [i.join() for i in list_processes]
    list_memory = [i.get() for i in list_memory]
    for memory_1 in list_memory:
        for memory_2 in memory_1:
            for key_model in memory_2.keys():
                result_models_info[key_model] = memory_2[key_model]
    result_models_info = dict(sorted(result_models_info.items(), key=lambda x: x[1][metric_for_best]))
    with open('sorted_models_info.json', 'w') as file:
        json.dump(obj=result_models_info, fp=file, indent=4, ensure_ascii=False)
    columns_y = pd.read_csv(initial['data_path']).columns[number_of_x::]
    for column_name in columns_y:
        for key, value in result_models_info.items():
            if column_name == value['name_y']:
                best_result_models_info[key] = value
                break


    with open('best_sorted_models_info.json', 'w') as file:
        json.dump(obj=best_result_models_info, fp=file, indent=4, ensure_ascii=False)

    import graphics
    graphics.graphics()

    with open('bool.txt', 'w') as file:
        file.write('End')









