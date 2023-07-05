
import pandas as pd

from effieciency_mf import train_test_split_special, calculate_sgd_efficiency, calculate_svd_efficiency
from memory_profiler import memory_usage


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped
def evaluate_sgd_and_svd_efficiency():

    sgd_iterations = 100
    sgd_factors = 40
    sgd_lr = .001
    sgd_reg = .02

    ratings_data_path = {
        '100_data': 'yelp_100_data/ratings.json',
        # '1000_data': 'yelp_1000_data/ratings.json',
        # '2000_data': 'yelp_2000_data/ratings.json',
        # '3000_data': 'yelp_3000_data/ratings.json',
        # '4000_data': 'yelp_4000_data/ratings.json',
        # '5000_data': 'yelp_5000_data/ratings.json',
    }

    # evaluation_result = pd.DataFrame(columns=['memory_usage', 'train_time', 'algo'])

    evaluation_result_sgd = pd.DataFrame(columns=['memory_usage', 'train_time', 'algo'])
    evaluation_result_svd = pd.DataFrame(columns=['memory_usage', 'train_time', 'algo'])

    for data_name, data_path in ratings_data_path.items():
        random_states = [
            73,
            # 57, 89, 26, 91, 34, 10, 64, 7, 51, 95, 13,
            # 42, 70, 18, 83, 29, 38, 76, 3, 68, 24, 49, 81,
            # 16, 61, 5, 28, 87, 33
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
                'n_factors': sgd_factors,
                'learning_rate': sgd_lr,
                'user_bias_reg': sgd_reg,
                'place_bias_reg': sgd_reg,
                'user_reg': sgd_reg,
                'place_reg': sgd_reg,
                'user_ids': user_ids,
                'place_ids': place_ids,
                'train_data': train_data,
                'test_data': test_data
            }
            wrapped = wrapper(
                calculate_sgd_efficiency,
                **parameters_sgd
            )
            mem_usage, train_time = memory_usage(wrapped, retval=True)
            mem_usage = max(mem_usage)

            evaluation_result_sgd = pd.concat([
                evaluation_result_sgd,
                pd.DataFrame({
                    'memory_usage': [mem_usage],
                    'train_time': [train_time],
                    'algo': ['sgd'],
                    'data_name': [data_name],
                    'random_state': [rs]
                })
            ], ignore_index=True)
            evaluation_result_sgd.to_csv('evaluation_result/efficiency_evaluation_result_sgd.csv', index=False)


            parameters_svd = {
                'train_data': train_data,
                'test_data': test_data,
            }
            wrapped = wrapper(
                calculate_svd_efficiency,
                **parameters_svd
            )
            mem_usage, train_time = memory_usage(wrapped, retval=True)
            mem_usage = max(mem_usage)

            evaluation_result_svd = pd.concat([
                evaluation_result_svd,
                pd.DataFrame({
                    'memory_usage': [mem_usage],
                    'train_time': [train_time],
                    'algo': ['svd'],
                    'data_name': [data_name],
                    'random_state': [rs]
                })
            ], ignore_index=True)
            evaluation_result_svd.to_csv('evaluation_result/efficiency_evaluation_result_svd.csv', index=False)

        print(f"Done evaluating efficiency for {data_name} data")

if __name__ == '__main__':
    evaluate_sgd_and_svd_efficiency()



