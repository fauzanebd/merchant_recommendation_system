import numpy as np
import pandas as pd
from FIX_MF import calculate_sgd_performance, train_test_split_special, calculate_svd_performance
import time
from memory_profiler import memory_usage


def find_best_sgd_parameters():
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

    best_params = {'test_rmse': np.inf, 'iterations': None, 'factors': None}

    res = pd.DataFrame(columns=['rs', 'n_iter', 'n_factors', 'train_rmse', 'test_rmse'])

    for rs in random_states:

        train_data, test_data = train_test_split_special(
            ratings_data,
            test_size=.2,
            random_state=rs
        )

        for k in latent_factors:
            print(f"rs: {rs}, k: {k}")
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

            for i, (tr_rmse, ts_rmse) in enumerate(zip(train_rmse, test_rmse)):
                # res = res.append(
                #     {'rs': rs, 'n_iter': iterations[i], 'n_factors': k, 'train_rmse': tr_rmse, 'test_rmse': ts_rmse},
                #     ignore_index=True)
                res = pd.concat(
                    [res, pd.DataFrame(
                        {'rs': rs, 'n_iter': iterations[i], 'n_factors': k, 'train_rmse': tr_rmse, 'test_rmse': ts_rmse},
                        index=[0])],
                    ignore_index=True
                )

                if ts_rmse < best_params['test_rmse']:
                    best_params = {'test_rmse': ts_rmse, 'iterations': iterations[i], 'factors': k}

    return best_params, res

def evaluate_best_parameter_sgd_performance(
    iterations,
    factors,
    learning_rate,
    regularization,
    user_ids,
    place_ids,
    train_data,
    test_data
):
    train_rmse, test_rmse = calculate_sgd_performance(
        iterations=iterations,
        train_data=train_data,
        test_data=test_data,
        user_ids=user_ids,
        place_ids=place_ids,
        learning_rate=learning_rate,
        user_bias_reg=regularization,
        place_bias_reg=regularization,
        user_reg=regularization,
        place_reg=regularization,
        n_factors=factors
    )

    return train_rmse[0], test_rmse[0]

def sgd_and_svd_evaluation():

    # Find best parameters for SGD
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
        columns=['data_name', 'rs', 'train_rmse', 'test_rmse', 'time_usage', 'algo'])

    print('Evaluating SGD and SVD...')
    for data_name, data_path in ratings_data_path.items():
        random_states = [
            73, 57, 89, 26, 91, 34, 10, 64, 7, 51, 95, 13,
            42, 70, 18, 83, 29, 38, 76, 3, 68, 24, 49, 81,
            16, 61, 5, 28, 87, 33
        ]
        ratings_data = pd.read_json(data_path)
        ratings_data.drop(columns=['ReviewID'], axis=1, inplace=True)
        uid_to_int = {uid: iid for iid, uid in enumerate(ratings_data['UserID'].unique())}
        pid_to_int = {pid: iid for iid, pid in enumerate(ratings_data['PlaceID'].unique())}
        int_to_uid = {iid: uid for iid, uid in enumerate(ratings_data['UserID'].unique())}
        int_to_pid = {iid: pid for iid, pid in enumerate(ratings_data['PlaceID'].unique())}
        ratings_data['UserID'] = ratings_data['UserID'].map(uid_to_int)
        ratings_data['PlaceID'] = ratings_data['PlaceID'].map(pid_to_int)

        user_ids = ratings_data['UserID'].unique()
        place_ids = ratings_data['PlaceID'].unique()
        for rs in random_states:
            train_data, test_data = train_test_split_special(
                ratings_data,
                test_size=.2,
                random_state=rs
            )
            print('Random state: {}'.format(rs))
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
            train_rmse, test_rmse = evaluate_best_parameter_sgd_performance(**parameters_sgd)
            end_time = time.time()
            evaluation_result = pd.concat([
                evaluation_result,
                pd.DataFrame({
                    'data_name': data_name,
                    'rs': rs,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'time_usage': end_time - start_time,
                    'algo': 'sgd'
                }, index=[0])
            ], ignore_index=True)

            parameters_svd = {
                'train_data': train_data,
                'test_data': test_data,
            }
            start_time = time.time()
            train_rmse, test_rmse = calculate_svd_performance(**parameters_svd)
            end_time = time.time()
            evaluation_result = pd.concat([
                evaluation_result,
                pd.DataFrame({
                    'data_name': data_name,
                    'rs': rs,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'time_usage': end_time - start_time,
                    'algo': 'svd'
                }, index=[0])
            ], ignore_index=True)
            evaluation_result.to_csv('evaluation_result/evaluation_result_sgd_and_svd.csv', index=False)
    print(f"Finished evaluating SGD and SVD")


if __name__ == "__main__":
    sgd_and_svd_evaluation()




