import numpy as np
import pandas as pd
from FIX_MF import calculate_sgd_performance, train_test_split_special, calculate_svd_performance
import time
from memory_profiler import memory_usage
import os
from multiprocessing import Pool
from functools import partial
import psutil
from evaluate_sgd_svd import evaluate_best_parameter_sgd_performance
import logging



def parameter_finder_worker_func(rs, k, iterations, learning_rate, regularization, user_ids, place_ids, ratings_data):
    logging.info('Running parameter_finder_worker_func with rs=%s, k=%s', rs, k)
    train_data, test_data = train_test_split_special(
        ratings_data,
        test_size=.2,
        random_state=rs
    )

    train_rmse, test_rmse = calculate_sgd_performance(
        iterations=iterations,
        train_data=train_data.copy(),
        test_data=test_data.copy(),
        user_ids=user_ids,
        place_ids=place_ids,
        learning_rate=learning_rate,
        user_bias_reg=regularization,
        place_bias_reg=regularization,
        user_reg=regularization,
        place_reg=regularization,
        n_factors=k
    )

    res = pd.DataFrame(columns=['rs', 'n_iter', 'n_factors', 'train_rmse', 'test_rmse'])
    best_params = {'test_rmse': np.inf, 'iterations': None, 'factors': None}

    for i, (tr_rmse, ts_rmse) in enumerate(zip(train_rmse, test_rmse)):
        res = pd.concat(
            [res, pd.DataFrame(
                {'rs': rs, 'n_iter': iterations[i], 'n_factors': k, 'train_rmse': tr_rmse, 'test_rmse': ts_rmse},
                index=[0])],
            ignore_index=True
        )

        if ts_rmse < best_params['test_rmse']:
            best_params = {'test_rmse': ts_rmse, 'iterations': iterations[i], 'factors': k}

    return best_params, res

def find_best_sgd_parameters():
    logging.info('Starting finding best_sgd_parameters')
    # Preprocessing and variables initialisation
    ratings_data = pd.read_json("yelp_1000_data/ratings.json")
    ratings_data.drop(columns=['ReviewID'], axis=1, inplace=True)
    uid_to_int = {uid: iid for iid, uid in enumerate(ratings_data['UserID'].unique())}
    pid_to_int = {pid: iid for iid, pid in enumerate(ratings_data['PlaceID'].unique())}
    int_to_uid = {iid: uid for iid, uid in enumerate(ratings_data['UserID'].unique())}
    int_to_pid = {iid: pid for iid, pid in enumerate(ratings_data['PlaceID'].unique())}
    ratings_data['UserID'] = ratings_data['UserID'].map(uid_to_int)
    ratings_data['PlaceID'] = ratings_data['PlaceID'].map(pid_to_int)

    user_ids = ratings_data['UserID'].unique()
    place_ids = ratings_data['PlaceID'].unique()

    random_states = [
        73, 57, 89, 26, 91, 34, 10, 64, 7, 51, 95, 13,
        42, 70, 18, 83, 29, 38, 76, 3, 68, 24, 49, 81,
        16, 61, 5, 28, 87, 33
    ]

    iterations = [10, 25, 50, 100, 200]
    latent_factors = [10, 20, 40, 80]
    learning_rate = .001
    regularization = .02

    parameters = [(rs, k) for rs in random_states for k in latent_factors]
    with Pool() as p:
        results = p.map(partial(parameter_finder_worker_func, iterations=iterations, learning_rate=learning_rate, regularization=regularization,
                                user_ids=user_ids, place_ids=place_ids, ratings_data=ratings_data), parameters)

    best_params_list, res_list = zip(*results)

    best_params = min(best_params_list, key=lambda x:x['test_rmse'])
    res = pd.concat(res_list, ignore_index=True)

    return best_params, res

def sgd_worker_func(args):
    logging.info('Starting sgd_worker_func with args=%s', args)
    rs, data_name, data_path, sgd_iterations, sgd_factors, sgd_learning_rate, sgd_regularization = args
    ratings_data = pd.read_json(data_path)
    ratings_data.drop(columns=['ReviewID'], axis=1, inplace=True)
    uid_to_int = {uid: iid for iid, uid in enumerate(ratings_data['UserID'].unique())}
    pid_to_int = {pid: iid for iid, pid in enumerate(ratings_data['PlaceID'].unique())}
    ratings_data['UserID'] = ratings_data['UserID'].map(uid_to_int)
    ratings_data['PlaceID'] = ratings_data['PlaceID'].map(pid_to_int)
    user_ids = ratings_data['UserID'].unique()
    place_ids = ratings_data['PlaceID'].unique()
    train_data, test_data = train_test_split_special(
        ratings_data,
        test_size=.2,
        random_state=rs
    )

    parameters_sgd = {
        'iterations': [sgd_iterations],
        'factors': sgd_factors,
        'learning_rate': sgd_learning_rate,
        'regularization': sgd_regularization,
        'user_ids': user_ids,
        'place_ids': place_ids,
        'train_data': train_data,
        'test_data': test_data
    }
    start_time = time.time()
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    train_rmse, test_rmse = evaluate_best_parameter_sgd_performance(**parameters_sgd)
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    end_time = time.time()
    mem_usage = end_mem - start_mem  # MB
    time_usage = end_time - start_time
    return data_name, rs, train_rmse, test_rmse, time_usage, 'sgd', mem_usage

def svd_worker_func(args):
    logging.info('Starting svd_worker_func with args=%s', args)
    rs, data_name, data_path = args
    ratings_data = pd.read_json(data_path)
    ratings_data.drop(columns=['ReviewID'], axis=1, inplace=True)
    uid_to_int = {uid: iid for iid, uid in enumerate(ratings_data['UserID'].unique())}
    pid_to_int = {pid: iid for iid, pid in enumerate(ratings_data['PlaceID'].unique())}
    ratings_data['UserID'] = ratings_data['UserID'].map(uid_to_int)
    ratings_data['PlaceID'] = ratings_data['PlaceID'].map(pid_to_int)
    train_data, test_data = train_test_split_special(
        ratings_data,
        test_size=.2,
        random_state=rs
    )

    parameters_svd = {
        'train_data': train_data,
        'test_data': test_data,
    }
    start_time = time.time()
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    train_rmse, test_rmse = calculate_svd_performance(**parameters_svd)
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    end_time = time.time()
    mem_usage = end_mem - start_mem  # MB
    time_usage = end_time - start_time
    return data_name, rs, train_rmse, test_rmse, time_usage, 'svd', mem_usage

def sgd_and_svd_evaluation():
    # Find best parameters for SGD
    logging.info('Starting sgd_and_svd_evaluation')
    print('Finding best parameters for SGD...')
    best_params, result_of_finding_best_params = find_best_sgd_parameters()
    result_of_finding_best_params.to_csv('evaluation_result/result_of_finding_best_params_for_sgd.csv', index=False)

    sgd_iterations = best_params['iterations']
    sgd_factors = best_params['factors']
    sgd_learning_rate = .001
    sgd_regularization = .02

    ratings_data_path = {
        '100_data': 'yelp_100_data/ratings.json',
        '1000_data': 'yelp_1000_data/ratings.json',
        '2000_data': 'yelp_2000_data/ratings.json',
        '3000_data': 'yelp_3000_data/ratings.json',
        '4000_data': 'yelp_4000_data/ratings.json',
        '5000_data': 'yelp_5000_data/ratings.json',
    }

    evaluation_result = pd.DataFrame(
        columns=['data_name', 'rs', 'train_rmse', 'test_rmse', 'memory_usage', 'time_usage', 'algo'])

    print('Evaluating SGD and SVD...')

    random_states = [
        73, 57, 89, 26, 91, 34, 10, 64, 7, 51, 95, 13,
        42, 70, 18, 83, 29, 38, 76, 3, 68, 24, 49, 81,
        16, 61, 5, 28, 87, 33
    ]

    args_sgd = [(rs, data_name, data_path, sgd_iterations, sgd_factors, sgd_learning_rate, sgd_regularization) for data_name, data_path in ratings_data_path.items() for rs in random_states]
    args_svd = [(rs, data_name, data_path) for data_name, data_path in ratings_data_path.items() for rs in random_states]

    with Pool() as p:
        results_sgd = p.map(sgd_worker_func, args_sgd)
        results_svd = p.map(svd_worker_func, args_svd)

    for result in results_sgd + results_svd:
        evaluation_result = pd.concat([
            evaluation_result,
            pd.DataFrame({
                'data_name': result[0],
                'rs': result[1],
                'train_rmse': result[2],
                'test_rmse': result[3],
                'time_usage': result[4],
                'algo': result[5],
                'memory_usage': result[6]
            }, index=[0])
        ], ignore_index=True)
    evaluation_result.to_csv('evaluation_result/evaluation_result_sgd_and_svd.csv', index=False)

    print(f"Finished evaluating SGD and SVD")
    logging.info('Finished evaluating SGD and SVD')

if __name__ == '__main__':
    # Setting up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    sgd_and_svd_evaluation()
