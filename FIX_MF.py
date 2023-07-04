
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import sys


def compute_sgd(
        shuffled_training_data,
        user_bias_reg, place_bias_reg, user_reg,
        place_reg, learning_rate, p, q,
        user_bias, place_bias, global_bias
):
    for user_id, place_id, rating_ui in shuffled_training_data.values:
        prediction = global_bias + user_bias[user_id] + place_bias[place_id]
        prediction += np.dot(p[user_id], q[place_id])
        error = rating_ui - prediction

        # Update biases
        user_bias[user_id] += learning_rate * (error - (user_bias_reg * user_bias[user_id]))
        place_bias[place_id] += learning_rate * (error - (place_bias_reg * place_bias[place_id]))

        # Update latent factors
        p_u = p[user_id]
        p[user_id] += learning_rate * ((error * q[place_id]) - (user_reg * p[user_id]))
        q[place_id] += learning_rate * ((error * p_u) - (place_reg * q[place_id]))

    return p, q, user_bias, place_bias


def train(
        training_rating_data: pd.DataFrame,
        user_ids,
        place_ids,
        user_bias_reg,
        place_bias_reg,
        user_reg,
        place_reg,
        number_of_users: int,
        number_of_places: int,
        learning_rate=.01,
        n_epochs=10,
        n_factors=10
):
    n_user = number_of_users
    n_place = number_of_places
    p = dict(zip(
        user_ids,
        np.random.normal(scale=1. / n_factors, size=(n_user, n_factors))
    ))
    q = dict(zip(
        place_ids,
        np.random.normal(scale=1. / n_factors, size=(n_place, n_factors))
    ))
    user_bias = dict(zip(
        user_ids,
        np.zeros(n_user)
    ))
    place_bias = dict(zip(
        place_ids,
        np.zeros(n_place)
    ))
    global_bias = np.mean(training_rating_data['Ratings'])
    p, q, user_bias, place_bias, global_bias = partial_train(
        training_rating_data, user_bias_reg, place_bias_reg,
        user_reg, place_reg, learning_rate, n_epochs,
        p, q, user_bias, place_bias, global_bias
    )
    return p, q, user_bias, place_bias, global_bias


def partial_train(
        training_rating_data, user_bias_reg, place_bias_reg,
        user_reg, place_reg, learning_rate, n_epochs,
        p, q, user_bias, place_bias, global_bias
):
    for _ in range(n_epochs):
        shuffled_training_data = training_rating_data.sample(frac=1).reset_index(drop=True)
        # print(f"length of training indices: {len(shuffled_training_data)}")
        p, q, user_bias, place_bias = compute_sgd(
            shuffled_training_data,
            user_bias_reg, place_bias_reg, user_reg,
            place_reg, learning_rate, p, q,
            user_bias, place_bias, global_bias
        )
    return p, q, user_bias, place_bias, global_bias


def train_test_split_special(df, test_size=0.2, random_state=42):
    # Initialize empty DataFrames for train and test data
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # Group data by user_id
    grouped = df.groupby('UserID')

    # For each user, split their data into train and test sets
    for _, group in grouped:
        if len(group) >= 2:  # Ensure the user has at least 2 ratings
            train_part, test_part = train_test_split(group, test_size=test_size, random_state=random_state)
            train_data = pd.concat([train_data, train_part], ignore_index=True)
            test_data = pd.concat([test_data, test_part], ignore_index=True)
        else:
            train_data = pd.concat([train_data, group], ignore_index=True)

    # if a place only appear in test data, move it to train data
    place_test_not_train = test_data.loc[
        ~test_data['PlaceID'].isin(train_data['PlaceID'].unique().tolist()), 'PlaceID'].unique().tolist()
    for place_id in place_test_not_train:
        train_data = pd.concat([train_data, test_data[test_data['PlaceID'] == place_id]], ignore_index=True)
        test_data.drop(test_data[test_data['PlaceID'] == place_id].index, inplace=True)
        test_data = test_data.reset_index(drop=True)

    return train_data, test_data


def predict(
        user_ids,
        place_ids,
        p,
        q,
        user_bias,
        place_bias,
        global_bias
):
    predictions = []
    for user_id, place_id in zip(user_ids, place_ids):
        prediction = global_bias + user_bias[user_id] + place_bias[place_id]
        prediction += p[user_id].dot(q[place_id].T)
        predictions.append(prediction)
    return predictions


def predict_single(
        user_id,
        place_id,
        p,
        q,
        user_bias,
        place_bias,
        global_bias
):
    prediction = global_bias + user_bias[user_id] + place_bias[place_id]
    prediction += p[user_id].dot(q[place_id].T)

    return prediction

def predict_svd(
        calc_pred_ratings_df,
        user_ids,
        place_ids,
):
    predictions = []
    for user_id, place_id in zip(user_ids, place_ids):
        prediction = calc_pred_ratings_df.loc[user_id, place_id]
        predictions.append(prediction)
    return predictions


def calculate_svd_performance(
        train_data,
        test_data,
):
    ratings_pivot = train_data.pivot_table(
        values="Ratings",
        index="UserID",
        columns="PlaceID"
    )
    avg_ratings = ratings_pivot.mean(axis=1)
    ratings_centered = ratings_pivot.sub(avg_ratings, axis=0)
    ratings_centered.fillna(0, inplace=True)

    U, sigma, Vt = svds(csr_matrix(ratings_centered))
    sigma = np.diag(sigma)
    U_sigma = np.dot(U, sigma)
    U_sigma_Vt = np.dot(U_sigma, Vt)
    uncentered_ratings = U_sigma_Vt + avg_ratings.values.reshape(-1, 1)
    calc_pred_ratings_df = pd.DataFrame(
        data=uncentered_ratings,
        index=ratings_pivot.index,
        columns=ratings_pivot.columns
    )

    train_predictions = predict_svd(
        calc_pred_ratings_df=calc_pred_ratings_df,
        user_ids=train_data['UserID'].tolist(),
        place_ids=train_data['PlaceID'].tolist()
    )
    train_rmse = np.sqrt(mean_squared_error(train_data['Ratings'].tolist(), train_predictions))

    test_predictions = predict_svd(
        calc_pred_ratings_df=calc_pred_ratings_df,
        user_ids=test_data['UserID'].tolist(),
        place_ids=test_data['PlaceID'].tolist()
    )
    test_rmse = np.sqrt(mean_squared_error(test_data['Ratings'].tolist(), test_predictions))

    return train_rmse, test_rmse


def calculate_sgd_performance(
        iterations,
        train_data,
        test_data,
        user_ids,
        place_ids,
        learning_rate,
        user_bias_reg,
        place_bias_reg,
        user_reg,
        place_reg,
        n_factors
):
    iterations.sort()
    train_mse = []
    test_mse = []
    train_rmse = []
    test_rmse = []
    iter_diff = 0

    n_train_user = len(user_ids)
    n_train_place = len(place_ids)

    for (i, n_iter) in enumerate(iterations):
        print(f"Iteration: {n_iter}")
        if i == 0:
            p, q, user_bias, place_bias, global_bias = train(
                training_rating_data=train_data,
                user_ids=user_ids,
                place_ids=user_ids,
                user_bias_reg=user_bias_reg,
                place_bias_reg=place_bias_reg,
                user_reg=user_reg,
                place_reg=place_reg,
                number_of_users=n_train_user,
                number_of_places=n_train_place,
                learning_rate=learning_rate,
                n_epochs=n_iter - iter_diff,
                n_factors=n_factors
            )
        else:
            p, q, user_bias, place_bias, global_bias = partial_train(
                training_rating_data=train_data,
                user_bias_reg=user_bias_reg,
                place_bias_reg=place_bias_reg,
                user_reg=user_reg,
                place_reg=place_reg,
                learning_rate=learning_rate,
                n_epochs=n_iter - iter_diff,
                p=p, q=q, user_bias=user_bias,
                place_bias=place_bias,
                global_bias=global_bias
            )

        train_predictions = predict(
            train_data['UserID'].tolist(),
            train_data['PlaceID'].tolist(),
            p, q, user_bias, place_bias, global_bias
        )
        test_predictions = predict(
            test_data['UserID'].tolist(),
            test_data['PlaceID'].tolist(),
            p, q, user_bias, place_bias, global_bias
        )

        train_mse += [mean_squared_error(train_data['Ratings'].tolist(), train_predictions)]
        test_mse += [mean_squared_error(test_data['Ratings'].tolist(), test_predictions)]
        train_rmse += [np.sqrt(train_mse[-1])]
        test_rmse += [np.sqrt(test_mse[-1])]
        iter_diff = n_iter

    return train_rmse, test_rmse




