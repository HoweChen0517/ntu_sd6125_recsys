import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import math
import os
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from model.dssm import DSSM
import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = ROOT_PATH = 'f:\\NTU Learn\\DATA MINING\\DMproject'

def data_process(data_path):
    data = pd.read_csv(data_path, sep='\t')
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data = data.sort_values(by='timestamp', ascending=True)
    train = data.iloc[:int(len(data)*0.8)].copy()
    test = data.iloc[int(len(data)*0.8):].copy()
    return train, test, data


def get_user_feature(data):

    def get_user_history(data, MAX_HIST_NUM=50):
        data_group = data[['userId', 'itemId', 'timestamp']].groupby('userId').apply(
            lambda x: x.sort_values('timestamp', ascending=False).head(MAX_HIST_NUM)).reset_index(drop=True)
        data_group = data_group.groupby('userId')['itemId'].apply(lambda x: '|'.join([str(i) for i in x])).reset_index()
        data_group.rename(columns={'itemId': 'user_hist'}, inplace=True)
        return data_group['user_hist']
    
    data_group = data[data['rating'] == 1]
    data_group = data_group[['userId', 'itemId']].groupby('userId').agg(list).reset_index()
    data_group['user_hist'] = get_user_history(data)
    data = pd.merge(data_group.drop('itemId', axis=1), data, on='userId')
    data_group = data[['userId', 'rating']].groupby('userId').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'user_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='userId')
    return data

def get_item_feature(data):
    data_group = data[['itemId', 'rating']].groupby('itemId').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'item_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='itemId')
    return data


def get_var_feature(data, col):
    key2index = {}

    def split(x):
        if type(x) == float:
            return [0]
        else:
            key_ans = x.split('|')
            for key in key_ans:
                if key not in key2index:
                    # Notice : input value 0 is a special "padding",\
                    # so we do not use 0 to encode valid feature for sequence input
                    key2index[key] = len(key2index) + 1
            return list(map(lambda x: key2index[x], key_ans))

    var_feature = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature)))
    max_len = max(var_feature_length)
    var_feature = pad_sequences(var_feature, maxlen=max_len, padding='post', )
    return key2index, var_feature, max_len


def get_test_var_feature(data, col, key2index, max_len):
    print("user_hist_list: \n")

    def split(x):
        if type(x) == float:
            return [0]
        else:
            key_ans = x.split('|')
            for key in key_ans:
                if key not in key2index:
                    # Notice : input value 0 is a special "padding",\
                    # so we do not use 0 to encode valid feature for sequence input
                    key2index[key] = len(key2index) + 1
            return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding='post')
    return test_hist


if __name__ == '__main__':
    # %%
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}:loading data...')
    data_set = 'ml-20m'
    data_path = os.path.join(ROOT_PATH, 'data', data_set, 'feature_df_0.txt')
    
    print('loading data...')
    train, test, data = data_process(data_path)
    train = get_user_feature(train)
    train = get_item_feature(train)

    # data = pd.read_parquet(r'F:\NTU Learn\DATA MINING\DMproject\data\ml-20m\feature_df_0.parquet')
    # data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    # data = data.sort_values(by='timestamp', ascending=True)
    # train = data.iloc[:math.ceil(len(data)*0.8)]
    # test = data.iloc[math.ceil(len(data)*0.8):]

    sparse_features = ['userId', 'itemId']
    dense_features = ['item_mean_rating', 'user_mean_rating']
    target = ['rating']

    user_sparse_features, user_dense_features = ['userId', ], ['user_mean_rating']
    item_sparse_features, item_dense_features = ['itemId', ], ['item_mean_rating']
    
    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        train[feat] = lbe.transform(train[feat])
        test[feat] = lbe.transform(test[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(train[dense_features])
    train[dense_features] = mms.transform(train[dense_features])
    # %%
    # 2.preprocess the sequence feature
    print(f'{time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}:proessing features...')
    genres_key2index, train_genres_list, genres_maxlen = get_var_feature(train, 'genres')
    user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')

    user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=8)
                            for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               user_dense_features]
    item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=8)
                            for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               item_dense_features]

    item_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=1000, embedding_dim=32),
                                                    maxlen=genres_maxlen, combiner='mean', length_name=None)]

    user_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('user_hist', vocabulary_size=26000, embedding_dim=32),
                                                    maxlen=user_maxlen, combiner='mean', length_name=None)]

    # %%
    # 3.generate input data for model
    user_feature_columns += user_varlen_feature_columns
    item_feature_columns += item_varlen_feature_columns

    # add user history as user_varlen_feature_columns
    train_model_input = {name: train[name] for name in sparse_features + dense_features}
    train_model_input["genres"] = train_genres_list
    train_model_input["user_hist"] = train_user_hist
    
    # %%
    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print(f'{time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}:cuda ready...')
        device = 'cuda:0'

    model = DSSM(user_feature_columns, item_feature_columns, task='binary', device=device, 
                 l2_reg_dnn=1e-3, l2_reg_embedding=1e-3, dnn_dropout=0.5)

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy'])
    # %%
    # 5. model training
    print(f'{time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}:model training...')
    model.fit(train_model_input, train[target].values, batch_size=1024, epochs=120, verbose=2, validation_split=0.2)
    # model.save
    # %%
    # 6.preprocess the test data
    print(f'{time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}: preprocessing test data...')
    test = pd.merge(test, train[['itemId', 'item_mean_rating']].drop_duplicates(), on='itemId', how='left').fillna(
        0.5)
    test = pd.merge(test, train[['userId', 'user_mean_rating']].drop_duplicates(), on='userId', how='left').fillna(
        0.5)
    test = pd.merge(test, train[['userId', 'user_hist']].drop_duplicates(), on='userId', how='left').fillna('1')
    test[dense_features] = mms.transform(test[dense_features])

    test_genres_list = get_test_var_feature(test, 'genres', genres_key2index, genres_maxlen)
    test_user_hist = get_test_var_feature(test, 'user_hist', user_key2index, user_maxlen)

    test_model_input = {name: test[name] for name in sparse_features + dense_features}
    test_model_input["genres"] = test_genres_list
    test_model_input["user_hist"] = test_user_hist
    # %%
    # 7.Evaluate
    print(f'{time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}:model evaluating...')
    eval_tr = model.evaluate(train_model_input, train[target].values)
    print(eval_tr)

    # %%
    # 8. model testing
    pred_ts = model.predict(test_model_input, batch_size=1024)
    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))

    # %%
    # 9.Embedding
    print("user embedding shape: ", model.user_dnn_embedding[:2])
    print("item embedding shape: ", model.item_dnn_embedding[:2])

    # %%
    # 10. get single tower
    dict_trained = model.state_dict()    # trained model
    trained_lst = list(dict_trained.keys())

    # user tower
    model_user = DSSM(user_feature_columns, [], task='binary', device=device)
    dict_user = model_user.state_dict()
    for key in dict_user:
        dict_user[key] = dict_trained[key]
    model_user.load_state_dict(dict_user)    # load trained model parameters of user tower
    user_feature_name = user_sparse_features + user_dense_features
    user_model_input = {name: test[name] for name in user_feature_name}
    user_model_input["user_hist"] = test_user_hist
    user_embedding = model_user.predict(user_model_input, batch_size=2000)
    print("single user embedding shape: ", user_embedding[:2])

    # item tower
    model_item = DSSM([], item_feature_columns, task='binary', device=device)
    dict_item = model_item.state_dict()
    for key in dict_item:
        dict_item[key] = dict_trained[key]
    model_item.load_state_dict(dict_item)  # load trained model parameters of item tower
    item_feature_name = item_sparse_features + item_dense_features
    item_model_input = {name: test[name] for name in item_feature_name}
    item_model_input["genres"] = test_genres_list
    item_embedding = model_item.predict(item_model_input, batch_size=2000)
    print("single item embedding shape: ", item_embedding[:2])

# %%
