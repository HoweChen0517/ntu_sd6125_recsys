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
import wandb
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_NOTEBOOK_NAME"] = 'f:\\NTU Learn\\DATA MINING\\DMproject\\src\\dssm_wandb_tune.py'

from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from model.dssm import DSSM
import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = 'f:\\NTU Learn\\DATA MINING\\DMproject'

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

def train(config=None):
    pass