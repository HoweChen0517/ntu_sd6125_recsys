import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
from collections import defaultdict
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.parquet as pq
import pyarrow as pa
import fastparquet
# import faiss
import joblib
warnings.filterwarnings('ignore')

ROOT_PATH = 'f:\\NTU Learn\\DATA MINING\\DMproject'

class CFinputs():
    
    def __init__(self, rating_matrix, user_index, item_index, data_set, batch_size=1000):

        """
        ratings_matrix: scipy.sparse.csr_matrix, user-item rating matrix
        user_index: list, user index
        item_index: list, item index
        data_set: str, 'ml-20m' or 'book_crossing'
        batch_size: int, default 1000
        """
        self.rating_matrix = rating_matrix
        self.user_index = user_index
        self.item_index = item_index
        self.data_set = data_set
        self.batch_size = batch_size

    def get_itemCF_sim_batch(self):
        item_user_matrix = self.rating_matrix.T
        items = self.item_index
        num_items = len(items)

        # 初始化一个 n*n 的空矩阵
        similarity_matrix = np.zeros((num_items, num_items))

        if self.data_set == 'ml-20m':
            output_file = os.path.join(ROOT_PATH, 'data', self.data_set, 'item_similarity_matrix.parquet')
        elif self.data_set == 'book_crossing':
            output_file = os.path.join(ROOT_PATH, 'data', self.data_set, 'item_similarity_matrix.parquet')

        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with tqdm(total=num_items // self.batch_size, desc="Calculating batch item similarities", unit="batch") as pbar:
                for start in range(0, num_items, self.batch_size):
                    end = min(start + self.batch_size, num_items)
                    batch_items = item_user_matrix[start:end, :]
                    
                    try:
                        batch_similarity = cosine_similarity(batch_items, item_user_matrix)
                    except ValueError as ve:
                        print(f"ValueError in cosine_similarity for batch {start}-{end}: {ve}")
                        continue
                    except MemoryError as me:
                        print(f"MemoryError: {me}")
                        break

                    similarity_matrix[start:end, :] = batch_similarity
                    
                    pbar.update(1)

            similarity_df = pd.DataFrame(similarity_matrix, index=items, columns=items)

            print('saving to parquet...')
            similarity_df.to_parquet(output_file)

        except OSError as e:
            print(f"OSError: Failed to create directory for {output_file}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        print(f"Item similarity matrix has been saved to {output_file}")


    # def get_userCF_sim_batch(self):
    #     user_item_matrix = self.rating_matrix
    #     users = self.user_index
    #     num_users = len(users)

    #     if self.data_set == 'ml-20m':
    #         output_file = os.path.join(ROOT_PATH, 'data', self.data_set, 'user_similarity.pkl')
    #     elif self.data_set == 'book_crossing':
    #         output_file = os.path.join(ROOT_PATH, 'data', self.data_set, 'user_similarity.pkl')

    #     try:
    #         os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
    #         with tqdm(total=num_users // self.batch_size, desc="Calculating user similarities", unit="batch") as pbar:
    #             try:
    #                 with open(output_file, mode='wb') as file:
    #                     for start in range(0, num_users, self.batch_size):
    #                         end = min(start + self.batch_size, num_users)
    #                         batch_users = user_item_matrix[start:end, :]
    #                         batch_index = [self.user_index[i] for i in range(start, end)]
                            
    #                         try:
    #                             # calculate cosine similarity between batch users and all users
    #                             similarity_matrix = cosine_similarity(batch_users, user_item_matrix)
    #                         except ValueError as ve:
    #                             print(f"ValueError in cosine_similarity for batch {start}-{end}: {ve}")
    #                             continue  # 跳过这个批次，继续下一个
    #                         except MemoryError as me:
    #                             print(f"MemoryError: {me}")
    #                             break  # end the loop if memory error occurs
                            
    #                         for idx, user_id in enumerate(batch_index):
    #                             user_similarity = {
    #                                 other_user_id: round(float(similarity), 4)
    #                                 for other_user_id, similarity in zip(users, similarity_matrix[idx])
    #                             }
                                
    #                             try:
    #                                 # save user similarity to file
    #                                 pickle.dump({user_id: user_similarity}, file)
    #                             except pickle.PicklingError as pe:
    #                                 print(f"PicklingError for user {user_id}: {pe}")
    #                                 continue  # skip this user and continue with the next one

    #                         pbar.update(1)

    #             except IOError as e:
    #                 print(f"IOError: Failed to open or write to file {output_file}: {e}")
    #             except Exception as e:
    #                 print(f"Unexpected error while writing to file: {e}")
        
    #     except OSError as e:
    #         print(f"OSError: Failed to create directory for {output_file}: {e}")
    #     except Exception as e:
    #         print(f"Unexpected error: {e}")
        
    #     print(f"User similarity results have been saved to {output_file}")

    def get_userCF_sim_batch(self):
        user_item_matrix = self.rating_matrix
        users = self.user_index
        num_users = len(users)

        similarity_matrix = np.zeros((num_users, num_users))

        if self.data_set == 'ml-20m':
            output_file = os.path.join(ROOT_PATH, 'data', self.data_set, 'user_similarity_matrix.parquet')
        elif self.data_set == 'book_crossing':
            output_file = os.path.join(ROOT_PATH, 'data', self.data_set, 'user_similarity_matrix.parquet')

        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with tqdm(total=num_users // self.batch_size, desc="Calculating batch user similarities", unit="batch") as pbar:
                for start in range(0, num_users, self.batch_size):
                    end = min(start + self.batch_size, num_users)
                    batch_users = user_item_matrix[start:end, :]
                    
                    try:
                        batch_similarity = cosine_similarity(batch_users, user_item_matrix)
                    except ValueError as ve:
                        print(f"ValueError in cosine_similarity for batch {start}-{end}: {ve}")
                        continue
                    except MemoryError as me:
                        print(f"MemoryError: {me}")
                        break

                    similarity_matrix[start:end, :] = batch_similarity
                    
                    pbar.update(1)

            similarity_df = pd.DataFrame(similarity_matrix, index=users, columns=users)

            print('saving to parquet...')
            similarity_df.to_parquet(output_file)

        except OSError as e:
            print(f"OSError: Failed to create directory for {output_file}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        print(f"User similarity matrix has been saved to {output_file}")