#%%
import numpy as np
import torch
import pandas as pd
import pickle
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix
from tqdm import tqdm
import gc
#%%
# load the item_sim data file
# itemCF_sim_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mNew\item_similarity_matrix.parquet'
itemCF_sim_file = r'F:/NTU Learn/DATA MINING/DMproject/data/ml-20m/item_similarity_matrix.parquet'
item_sim = pd.read_parquet(itemCF_sim_file)
#%%
# retrieve all the userId & movieId
# ratings_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mNew\ratings.csv'
ratings_file = r'F:/NTU Learn/DATA MINING/DMproject/data/ml-20m/ratings.csv'
# movies_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mNew\movies.csv'
movies_file = r'F:/NTU Learn/DATA MINING/DMproject/data/ml-20m/movies.csv'
ratings = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)

all_users = ratings['userId'].unique()
all_items = movies['movieId'].unique()
print('ratings shape: ', ratings.shape)
print('movies shape: ', movies.shape)
print("all_users: ", len(all_users))
print("all_items: ", len(all_items))
#%%
def convert_df_to_sparse_matrix(rating_df, flag='score'):

    """
    Input:
    flag: indicator for whether to use score or binary
    ----
    Output:
    sparse_matrix: scipy.sparse.lil_matrix, user-item matrix
    """

    user_ids = rating_df['userId'].unique()
    item_ids = rating_df['itemId'].unique()

    user_index_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_index_map = {item_id: idx for idx, item_id in enumerate(item_ids)}

    num_users = len(user_ids)
    num_items = len(item_ids)

    if flag == 'score':
        sparse_matrix = lil_matrix((num_users, num_items))
    elif flag == 'binary':
        sparse_matrix = lil_matrix((num_users, num_items), dtype=np.int8)

    for _, row in tqdm(rating_df.iterrows(), total=rating_df.shape[0], desc="Processing ratings"):
        user_id = row['userId']
        item_id = row['itemId']
        score = row['rating']

        user_index = user_index_map[user_id]
        item_index = item_index_map[item_id]

        if flag == 'score':
            sparse_matrix[int(user_index), int(item_index)] = int(score)
        elif flag == 'binary':
            sparse_matrix[int(user_index), int(item_index)] = 1

    sparse_matrix = sparse_matrix.tocoo()

    return list(user_ids), list(item_ids), sparse_matrix

# Convert the ratings DataFrame to a sparse user-item matrix
all_users, all_items, user_item_matrix = convert_df_to_sparse_matrix(ratings)
print("user_item_matrix shape: ", user_item_matrix.shape)
#%%
# the function to predict user's rating for a movie
def predict_ratings_matrix_in_batches(user_item_matrix, item_sim, batch_size=32, k=100):
    # get the number of users and items
    num_users, num_items = user_item_matrix.shape

    # 将稀疏矩阵转换为稠密矩阵
    user_item_dense = user_item_matrix.toarray()
    predictions = np.zeros((num_users, num_items), dtype=np.float32)

    for start in tqdm(range(0, num_items, batch_size), desc="Predicting ratings in batches"):
        end = min(start + batch_size, num_items)
        # Process items in batches to reduce memory usage
        batch_items = range(start, end)

        for item_id in batch_items:
            if item_id not in item_sim.index:
                continue
            # select all similar items in the item_sim matrix with the specific item identified by item_id
            similar_items = item_sim.loc[item_id,:].to_dict()

            # select the top-k similar items
            similar_items = sorted(similar_items.items(), key=lambda x: x[1], reverse=True)[:k]
            similar_item_ids, similarities = zip(*similar_items)

            # Use matrix operations to calculate the predicted rating for the current item
            similar_item_ids = np.array(similar_item_ids, dtype=np.int32)
            similarities = np.array(similarities, dtype=np.float32)

            # computing the rating
            for user_id in range(num_users):
                rated_items = user_item_dense[user_id, similar_item_ids]
                non_zero_indices = np.nonzero(rated_items)[0]
                if len(non_zero_indices) == 0: # if the user did not rate any similar item
                    predictions[user_id, item_id] = np.nan
                    continue

                rated_items = rated_items[non_zero_indices] # retrieve the user's all rated similar items
                similarities_subset = similarities[non_zero_indices] # retrieve these items' similarity

                numerator = np.dot(rated_items, similarities_subset) # rate × sim
                denominator = similarities_subset.sum() # sum(sim)

                predictions[user_id, item_id] = numerator / denominator if denominator != 0 else np.nan

        # clear the memory
        gc.collect()

    return predictions
#%%
# Calculate all users' predicted ratings for all items in a batch
predicted_ratings = predict_ratings_matrix_in_batches(user_item_matrix, item_sim)

# Transfer the predicted ratings back to CPU
# predicted_ratings = predicted_ratings_gpu.cpu().numpy()

# Use dictionary to save rating result
predictions_dict = {}
for user_idx, user_id in enumerate(all_users):
    predictions_dict[user_id] = {}
    for item_idx, item_id in enumerate(all_items):
        if not np.isnan(predicted_ratings[user_idx, item_idx]):
            predictions_dict[user_id][item_id] = predicted_ratings[user_idx, item_idx]
#%%
# save the rating to a file
# output_file_path_prediction = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\_ml\predicted_ratings_all_itemCF.pkl'
output_file_path_prediction = r'F:/NTU Learn/DATA MINING/DMproject/output/predicted_ratings_all_itemCF.pkl'
with open(output_file_path_prediction, 'wb') as output_file_ra:
    pickle.dump(predictions_dict, output_file_ra)

print("save successfully for rating")
#%%
# create a recommendation dict for user, sorted by rating
recommendations_dict = {}
for user_id, item_ratings in predictions_dict.items():
    sorted_items = sorted(item_ratings.items(), key=lambda x: x[1], reverse=True)
    recommendations_dict[user_id] = [item[0] for item in sorted_items]
#%%
# Save the recommendations dict to a file
# output_file_path_recommendation = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\_ml\user_recommendation_itemCF.pkl'
output_file_path_recommendation = r'F:/NTU Learn/DATA MINING/DMproject/output/user_recommendation_itemCF.pkl'
with open(output_file_path_recommendation, 'wb') as output_file_re:
    pickle.dump(recommendations_dict, output_file_re)

print("save successfully for recommendation")