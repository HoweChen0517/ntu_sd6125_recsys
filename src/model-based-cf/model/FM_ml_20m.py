import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader
from surprise import KNNWithMeans, SVD, NMF
from surprise import dump
from surprise.model_selection import cross_validate, GridSearchCV
from surprise.accuracy import rmse, mae
from sklearn.preprocessing import MinMaxScaler
import json
import logging
import pickle

# 设置根目录
ROOT_PATH = 'f:\\NTU Learn\\DATA MINING\\DMproject'

# 数据预处理
def data_preprocessing(df, data_set):
    # if data_set == 'ml-20m':
    #     df = df.rename(columns={'userId':'userId', 'movieId':'itemId', 'rating':'rating'})
    #     min_max_scaler = MinMaxScaler(feature_range=(0.5, 5))
    #     df['rating'] = min_max_scaler.fit_transform(df['rating'].values.reshape(-1, 1))
    # elif data_set == 'book_crossing':
    #     with open('data/book_crossing/map_dict.json', 'r') as f:
    #         map_dict = json.load(f)
    #     df['itemId'] = df['ISBN'].apply(lambda x: map_dict[x])
    #     df = df.rename(columns={'User-ID':'userId', 'Book-Rating':'rating'})
    #     min_max_scaler = MinMaxScaler(feature_range=(0.5, 5))
    #     df['rating'] = min_max_scaler.fit_transform(df['rating'].values.reshape(-1, 1))
    #     df = df.drop(columns=['ISBN'])
    return df

def load_train_test_data(data_set):
    # Load pre-split train and test sets
    train_file = os.path.join(ROOT_PATH, 'data', data_set, 'ml_20m_df_train.parquet')
    test_file = os.path.join(ROOT_PATH, 'data', data_set, 'ml_20m_df_test.parquet')

    df_train = pd.read_parquet(train_file)
    df_test = pd.read_parquet(test_file)
    
    reader = Reader(line_format='user item rating', sep=',')
    
    # Prepare the train and test datasets for Surprise
    train_data = Dataset.load_from_df(df_train[['userId', 'itemId', 'rating']], reader)
    test_data = Dataset.load_from_df(df_test[['userId', 'itemId', 'rating']], reader)
    
    return train_data, test_data

# load data
def load_data(data_set):
    # data_path = os.path.join(ROOT_PATH, 'data', data_set, 'ratings.csv')
    df = pd.read_csv(data_path)
    data_path = os.path.join(ROOT_PATH, 'data', data_set, 'ratings.csv')
    df = pd.read_csv(data_path)
    df = data_preprocessing(df, data_set)
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_df(df[['userId', 'itemId', 'rating']], reader)
    return data
# tune parameters
def tune_parameters(data, algo_class, param_grid, cv=3):
    gs = GridSearchCV(algo_class, param_grid, measures=['rmse', 'mae'], cv=cv)
    gs.fit(data)
    return gs.best_params['rmse'], gs.best_estimator['rmse']
# train model
def train_model(data, algo):
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    return algo
# model evaluation
# def evaluate_model(algo, data):
#     results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#     logging.info(results)
#     return results
def evaluate_model(algo, test_data):
    # Convert test_data into a Surprise test set format
    testset = test_data.build_full_trainset().build_testset()
    
    # Evaluate on test data
    predictions = algo.test(testset)
    rmse_score = rmse(predictions)
    mae_score = mae(predictions)
    results = {'test_rmse': rmse_score, 'test_mae': mae_score}
    logging.info(results)

    return results
# save model and results
def save_model_and_results(model, results, model_path, results_path):

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    serializable_results = {key: value for key, value in results.items() if isinstance(value, (int, float, str, list, dict))}

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)

# 主程序
def main(data_set='ml-20m'):
    data_set = data_set
    # 设置日志记录
    logging.basicConfig(level=logging.INFO, filename=os.path.join(ROOT_PATH,'log','FM',f'{data_set}_FM_training_1024.log'), filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

    model_filename = os.path.join(ROOT_PATH,'output','model',f'{data_set}_collaborative_filtering_model.pickle')
    results_filename = os.path.join(ROOT_PATH,'output','result',f'{data_set}_vector_based_CF_results.json')
    train_data, test_data = load_train_test_data(data_set)
    # forming grid search parameters for different algorithms
    algorithms = {
        'SVD': {
            'algo': SVD,
            'params': {
                'n_factors': [50, 100, 150],
                'n_epochs': [20, 30],
                'lr_all': [0.005, 0.01],
                'reg_all': [0.02, 0.1]
            }
        },
        'NMF': {
            'algo': NMF,
            'params': {
                'n_factors': [15, 20, 25],
                'n_epochs': [50, 100],
            }
        },
        'KNNWithMeans': {
            'algo': KNNWithMeans,
            'params': {
                'k': [10, 20, 30],
                'sim_options': {'name': ['cosine', 'pearson'], 'user_based': [True, False]}
            }
        }
    }
    # 保存最佳模型的信息
    best_model_info = {}
    # 对每个算法进行参数调优
    for algo_name, algo_data in algorithms.items():
        logging.info(f"begin tuning parameters for {data_set}: {algo_name}")
        best_params, best_algo = tune_parameters(train_data, algo_data['algo'], algo_data['params'])
        logging.info(f"{data_set}_best_params: {best_params}")
        logging.info(f"start training: {algo_name}")
        model = train_model(train_data, best_algo)
        
        logging.info(f"start evaluation: {algo_name}")

        print('-'*20+f"start evaluation: {algo_name}"+'-'*20)
        results = evaluate_model(model, test_data)
        # if the current model is better than the previous best model, update the best model info
        if not best_model_info or best_model_info['best_score'] > results['test_rmse'].mean():
            best_model_info = {
                'algorithm': algo_name,
                'best_score': results['test_rmse'].mean(),
                'best_params': best_params,
                'best_estimator': best_algo
            }
    # save the best model and results
    save_model_and_results(
        best_model_info['best_estimator'],
        best_model_info,
        model_filename,
        results_filename
    )
    logging.info(f"{data_set}_model is trained and saved at: {model_filename}")
    logging.info(f"{data_set}_model evaluation is saved at: {results_filename}")

if __name__ == "__main__":
    main(data_set='ml-20m')