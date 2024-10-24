import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
from collections import defaultdict
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import warnings
import os
import json
from tqdm import tqdm
warnings.filterwarnings('ignore')

ROOT_PATH = 'f:\\NTU Learn\\DATA MINING\\DMproject'

def read_data(data_set='ml-20m', sep=',',map_dict_path=None):
    
    """
    data_path: str, path to the data file
    sep: str, default ','
    header: int or list of ints, default 'infer'
    names: list, default None
    usecols: list, default None
    dtype: dict, default None
    """
    
    data_path = os.path.join(ROOT_PATH, 'data', data_set, 'ratings.csv')

    data = pd.read_csv(data_path, sep=sep)
    
    data = data.loc[:, ['userId', 'itemId', 'rating']]
    # if data_set == 'ml-20m':
    #     data = data.rename(columns={'userId':'userId', 'movieId':'itemId', 'rating':'rating', 'timestamp':'timestamp'})
    # elif data_set == 'book_crossing':
    #     map_dict_path = 'data/book_crossing/map_dict.json'
    #     with open(map_dict_path, 'r') as f:
    #         map_dict = json.load(f)
    #     data = data.rename(columns={'User-ID':'userId', 'ISBN':'itemId', 'Book-Rating':'rating'})
    #     data['userId'] = data['userId'].astype(int)
    #     data['itemId'] = data['itemId'].apply(lambda x: map_dict[x]).astype(int)
    #     data['rating'] = data['rating'].astype(int)
    # else:
    #     data = data

    return data


class train_test_split():

    def __init__(self, data, test_size, random_state, k=None):

        """
        data: pd.DataFrame, includes columns ['userId', 'itemId', 'rating']
        test_size: float
        random_state: int
        """
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.k = k

    def train_test_split_random(self, data, test_size, random_state):

        trn_data, val_data, _, _ = sklearn_train_test_split(data, data, test_size=test_size, random_state=random_state)

        return trn_data, val_data
    
    def train_test_split_timeorder(self, data, test_size):

        if 'timestamp' not in data.columns:
            raise ValueError("The input data does not contain timestamp column.")
        
        else:
            data_sorted = data.sort_values(by = ['userId','timestamp'])

            trn_data = pd.DataFrame(columns=data_sorted.columns)
            val_data = pd.DataFrame(columns=data_sorted.columns)

            for user_id, group in data_sorted.groupby('userId'):

                train_size = int(len(group) * (1-test_size))

                user_trn = group.iloc[:train_size]
                user_val = group.iloc[train_size:]

                trn_data = pd.concat([trn_data, user_trn], ignore_index=True)
                val_data = pd.concat([val_data, user_val], ignore_index=True)

            return trn_data, val_data

    def train_test_split_leavekout(self, ratings, k=1):

        """
        k: number of items to leave out for validation
        """

        ratings_sorted = ratings.sort_values(by = ['userId','timestamp'])

        trn_data = pd.DataFrame(columns=ratings_sorted.columns)
        val_data = pd.DataFrame(columns=ratings_sorted.columns)

        for user_id, group in ratings_sorted.groupby('userId'):

            trn_data = pd.concat([trn_data, group.iloc[:-k]], ignore_index=True)
            val_data = pd.concat([val_data, group.iloc[-k:]], ignore_index=True)

        return trn_data, val_data

class data_transformation():
    def __init__(self, data, flag='score'):
        self.data = data
        self.flag = flag

    def convert_df_to_dict(self, data):
            
            """
            data: pd.DataFrame, includes columns ['userId', 'itemId', 'rating'], either training or validation data
            """
    
            user_items_score = data.groupby('userId').apply(lambda x: dict(zip(x['itemId'], x['rating']))).to_dict()
            user_items = data.groupby('userId')['itemId'].apply(list).to_dict()
    
            return user_items_score, user_items
    
    # def convert_df_to_sparse_matrix(self, data, flag):

    #     """
    #     Input:
    #     data: record data of ratings
    #     flag: indicator for whether to use score or binary
    #     ----
    #     Output:
    #     sparse_matrix: scipy.sparse.lil_matrix, user-item matrix
    #     """

    #     user_ids = data['userId'].unique()
    #     item_ids = data['itemId'].unique()

    #     user_index_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    #     item_index_map = {item_id: idx for idx, item_id in enumerate(item_ids)}

    #     num_users = len(user_ids)
    #     num_items = len(item_ids)

    #     if flag == 'score':
    #         sparse_matrix = lil_matrix((num_users, num_items))
    #     elif flag == 'binary':
    #         sparse_matrix = lil_matrix((num_users, num_items), dtype=np.int8)

    #     for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing ratings"):
    #         user_id = row['userId'].astype(int)
    #         item_id = row['itemId'].astype(int)
    #         score = row['rating']

    #         user_index = user_index_map[user_id]
    #         item_index = item_index_map[item_id]

    #         if flag == 'score':
    #             sparse_matrix[user_index, item_index] = score
    #         elif flag == 'binary':
    #             sparse_matrix[user_index, item_index] = 1

    #     rating_matrix = sparse_matrix.tocsr()

    #     return list(user_ids), list(item_ids), rating_matrix
    
    def convert_rating_list_to_matrix(self, data, num_users, num_items, flag):

        """
        Input:
        data: record data of ratings
        flag: indicator for whether to use score or binary
        ----
        Output:
        np.array, user-item matrix
        """

        user_ids = data['userId'].unique()
        item_ids = data['itemId'].unique()

        user_index_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_index_map = {item_id: idx for idx, item_id in enumerate(item_ids)}

        rating_matrix = np.zeros((num_users, num_items))


        for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing ratings"):
            user_id = row['userId'].astype(int)
            item_id = row['itemId'].astype(int)
            score = row['rating']

            user_index = user_index_map[user_id]
            item_index = item_index_map[item_id]

            if flag == 'score':
                rating_matrix[user_index, item_index] = score
            elif flag == 'binary':
                rating_matrix[user_index, item_index] = 1

        return rating_matrix