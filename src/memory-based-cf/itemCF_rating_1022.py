#%%
import numpy as np
import torch
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from tqdm import tqdm
import os
import gc

ROOT_PATH = 'f:\\NTU Learn\\DATA MINING\\DMproject'
#%%
# load the item_sim data file
data_set = 'ml-20m'
itemCF_sim_file = os.path.join(ROOT_PATH, 'data', data_set, 'item_similarity_matrix.parquet')
# itemCF_sim_file = r'/home/msds/cli034/DataMiningproject/ml-20mNew/item_similarity_matrix.parquet'
# item_sim = pd.read_parquet(itemCF_sim_file).to_dict()
item_sim = pd.read_parquet(itemCF_sim_file)
#%%
# retrieve all the userId & movieId
ratings_file = os.path.join(ROOT_PATH, 'data', data_set, 'ratings.csv')
# ratings_file = r'/home/msds/cli034/DataMiningproject/ml-20mNew/ratings.csv'
# movies_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mNew\movies.csv'
# movies_file = r'/home/msds/cli034/DataMiningproject/ml-20mNew/movies.csv'
ratings = pd.read_csv(ratings_file)
# movies = pd.read_csv(movies_file)

all_users = ratings['userId'].unique()
all_items = ratings['itemId'].unique()
#%%
# use sparse matrix to load user_item data
user_item_matrix = coo_matrix((ratings['rating'], (ratings['userId'], ratings['itemId']))).tocsr()
#%%
# the function to predict user's rating for a movie
def predict_ratings_matrix_in_batches(user_item_matrix, item_sim, batch_size=4, k=100):
    # get the number of users and items
    num_users, num_items = user_item_matrix.shape

    # Transfer user-item matrix to GPU in a sparse format using PyTorch,
    # user_item_values_gpu = torch.tensor(user_item_matrix.data, dtype=torch.float32, device='cuda')
    # user_item_indices_gpu = torch.tensor(np.vstack((user_item_matrix.row, user_item_matrix.col)), dtype=torch.long, device='cuda')
    # user_item_sparse_gpu = torch.sparse_coo_tensor(user_item_indices_gpu, user_item_values_gpu, size=(num_users, num_items), device='cuda')
    # predictions = torch.zeros((num_users, num_items), dtype=torch.float32, device='cuda')
    # Create empty predictions array on CPU to avoid GPU memory issues
    predictions = np.memmap('predicted_ratings.dat', dtype=np.float32, mode='w+', shape=(num_users, num_items))

    for start in tqdm(range(0, num_items, batch_size), desc="Predicting ratings in batches"):
        end = min(start + batch_size, num_items)
        # Process items in batches to reduce memory usage
        batch_items = range(start, end)

        for item_id in batch_items:
            # if item_id not in item_sim:
            if item_id not in item_sim.index:
                continue
            # select all similar items in the item_sim matrix with the specific item identified by item_id
            # similar_items = item_sim[item_id]
            similar_items = item_sim.loc[item_id]

            # select the top-k similar items
            # similar_items = sorted(similar_items.items(), key=lambda x: x[1], reverse=True)[:k]
            # similar_item_ids, similarities = zip(*similar_items)
            similar_items = similar_items.sort_values(ascending=False)[:k]
            similar_item_ids = similar_items.index.values
            similarities = similar_items.values

            # # Use matrix operations to calculate the predicted rating for the current item
            # similar_item_ids_gpu = torch.tensor(similar_item_ids, dtype=torch.long, device='cuda')
            # similarities_gpu = torch.tensor(similarities, dtype=torch.float32, device='cuda')

            # computing the rating
            for user_id in range(num_users):
                # rated_items = user_item_sparse_gpu[user_item_indices_gpu[0] == user_id].to_dense()[similar_item_ids_gpu]
                # Move only the required data to GPU
                user_item_values_gpu = torch.tensor(user_item_matrix.getrow(user_id).toarray().flatten(),
                                                    dtype=torch.float32, device='cuda')
                similar_item_ids_gpu = torch.tensor(similar_item_ids, dtype=torch.long, device='cuda')
                similarities_gpu = torch.tensor(similarities, dtype=torch.float32, device='cuda')

                rated_items = user_item_values_gpu[similar_item_ids_gpu]

                non_zero_indices = rated_items.nonzero(as_tuple=True)[0]
                if len(non_zero_indices) == 0: # if the user did not rate any similar item
                    predictions[user_id, item_id] = np.nan
                    torch.cuda.empty_cache()
                    continue

                rated_items = rated_items[non_zero_indices] # retrieve the user's all rated similar items
                similarities_subset = similarities_gpu[non_zero_indices] # retrieve these items' similarity

                numerator = torch.dot(rated_items, similarities_subset) # rate × sim
                denominator = similarities_subset.sum() # sum(sim)

                # predictions[user_id, item_id] = numerator / denominator if denominator != 0 else np.nan
                predictions[user_id, item_id] = numerator.cpu().item() / denominator.cpu().item() if denominator != 0 else float(
                    'nan')
                torch.cuda.empty_cache()

        # clear the memory
        gc.collect()

    return predictions
#%%
# # Calculate all users' predicted ratings for all items in a batch
# predicted_ratings_gpu = predict_ratings_matrix_in_batches(user_item_matrix, item_sim)
#
# # Transfer the predicted ratings back to CPU
# predicted_ratings = predicted_ratings_gpu.cpu().numpy()
predicted_ratings = predict_ratings_matrix_in_batches(user_item_matrix, item_sim)

#%%
# Use dictionary to save rating result
predictions_dict = {}
for user_idx, user_id in enumerate(all_users):
    predictions_dict[user_id] = {}
    for item_idx, item_id in enumerate(all_items):
        if not np.isnan(predicted_ratings[user_idx, item_idx]):
            predictions_dict[user_id][item_id] = predicted_ratings[user_idx, item_idx]
#%%
# save the rating to a file
output_file_path_prediction = os.pawth.join(ROOT_PATH, 'output', data_set, 'result', 'predicted_ratings_all_itemCF.pkl')
# output_file_path_prediction = r'/home/msds/cli034/DataMiningproject/CF_ml/predicted_ratings_all_itemCF.pkl'
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
output_file_path_recommendation = os.pawth.join(ROOT_PATH, 'output', data_set, 'result', 'user_recommendation_itemCF.pkl')
# output_file_path_recommendation = r'/home/msds/cli034/DataMiningproject/CF_ml/user_recommendation_itemCF.pkl'
with open(output_file_path_recommendation, 'wb') as output_file_re:
    pickle.dump(recommendations_dict, output_file_re)

print("save successfully for recommendation")