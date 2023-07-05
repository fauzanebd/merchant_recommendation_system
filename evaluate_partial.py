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


def get_num_processes(data_size):
    # Change these numbers according to your resources and requirements
    if data_size <= 100:
        return 100
    elif data_size <= 1000:
        return 10
    elif data_size <= 2000:
        return 6
    elif data_size <= 3000:
        return 5
    else:  # data_size > 3000
        return 4


def svd_evaluation():
    # Find best parameters for SGD
    logging.info('Starting svd_evaluation')

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

    print('Evaluating SVD...')

    random_states = [
        73, 57, 89, 26, 91, 34, 10, 64, 7, 51, 95, 13,
        42, 70, 18, 83, 29, 38, 76, 3, 68, 24, 49, 81,
        16, 61, 5, 28, 87, 33
    ]

    for data_name, data_path in sorted(ratings_data_path.items(), key=lambda x: int(x[0].split('_')[0])):
        data_size = int(data_name.split('_')[0])  # get the size from the data_name
        num_processes = get_num_processes(data_size)
        args_svd = [(rs, data_name, data_path) for rs in random_states]
        if num_processes == 100:
            with Pool() as p:
                results_svd = p.map(svd_worker_func, args_svd)
        else:
            with Pool(processes=num_processes) as p:
                results_svd = p.map(svd_worker_func, args_svd)

        for result in results_svd:
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
            evaluation_result.to_csv('evaluation_result/evaluation_result_svd.csv', index=False)

    print(f"Finished evaluating SGD and SVD")
    logging.info('Finished evaluating SGD and SVD')


if __name__ == '__main__':
    # Setting up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    svd_evaluation()

