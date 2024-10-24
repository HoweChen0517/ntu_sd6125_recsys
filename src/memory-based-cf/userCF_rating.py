import numpy as np
import torch
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from tqdm import tqdm
import gc
#%%
# load the item_sim data file
# itemCF_sim_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mNew\user_similarity_matrix.parquet'
userCF_sim_file = r'/home/msds/cli034/DataMiningproject/ml-20mNew/user_similarity_matrix.parquet'
user_sim = pd.read_parquet(userCF_sim_file).to_dict()
#%%
# retrieve all the userId & movieId
# ratings_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mNew\ratings.csv'
ratings_file = r'/home/msds/cli034/DataMiningproject/ml-20mNew/ratings.csv'
# movies_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mNew\movies.csv'
movies_file = r'/home/msds/cli034/DataMiningproject/ml-20mNew/movies.csv'
ratings = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)

all_users = ratings['userId'].unique()
all_items = movies['movieId'].unique()
#%%
# use sparse matrix to load user_item data
user_item_matrix = coo_matrix((ratings['rating'], (ratings['userId'], ratings['itemId'])))
#%%
# the function to predict user's rating for a movie
def predict_ratings_matrix_in_batches(user_item_matrix, user_sim, batch_size=32, k=100):
    # get the number of users and items
    num_users, num_items = user_item_matrix.shape

    # Transfer user-item matrix to GPU in a sparse format using PyTorch,
    user_item_values_gpu = torch.tensor(user_item_matrix.data, dtype=torch.float32, device='cuda')
    user_item_indices_gpu = torch.tensor(np.vstack((user_item_matrix.row, user_item_matrix.col)), dtype=torch.long, device='cuda')
    user_item_sparse_gpu = torch.sparse_coo_tensor(user_item_indices_gpu, user_item_values_gpu, size=(num_users, num_items), device='cuda')
    predictions = torch.zeros((num_users, num_items), dtype=torch.float32, device='cuda')

    for start in tqdm(range(0, num_users, batch_size), desc="Predicting ratings in batches"):
        end = min(start + batch_size, num_items)
        # Process users in batches to reduce memory usage
        batch_users = range(start, end)

        for user_id in batch_users:
            if user_id not in user_sim:
                continue
            # select all similar users in the user_sim matrix with the specific user identified by user_id
            similar_users = user_sim[user_id]

            # select the top-k similar users
            similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:k]
            similar_user_ids, similarities = zip(*similar_users)

            # Use matrix operations to calculate the predicted rating for the current item
            similar_user_ids_gpu = torch.tensor(similar_user_ids, dtype=torch.long, device='cuda')
            similarities_gpu = torch.tensor(similarities, dtype=torch.float32, device='cuda')

            # computing the rating
            for item_id in range(num_items):
                rated_users = user_item_sparse_gpu[similar_user_ids_gpu, item_id].to_dense()
                non_zero_indices = rated_users.nonzero(as_tuple=True)[0]
                if len(non_zero_indices) == 0: # if no similar user rated the item
                    predictions[user_id, item_id] = np.nan
                    continue

                rated_users = rated_users[non_zero_indices] # retrieve all similar users rate the item
                similarities_subset = similarities_gpu[non_zero_indices] # retrieve these users' similarity

                numerator = torch.dot(rated_users, similarities_subset) # rate × sim
                denominator = similarities_subset.sum() # sum(sim)

                predictions[user_id, item_id] = numerator / denominator if denominator != 0 else np.nan

        # clear the memory
        gc.collect()

    return predictions
#%%
# Calculate all users' predicted ratings for all items in a batch
predicted_ratings_gpu = predict_ratings_matrix_in_batches(user_item_matrix, user_sim)

# Transfer the predicted ratings back to CPU
predicted_ratings = predicted_ratings_gpu.cpu().numpy()

# Use dictionary to save rating result
predictions_dict = {}
for user_idx, user_id in enumerate(all_users):
    predictions_dict[user_id] = {}
    for item_idx, item_id in enumerate(all_items):
        if not np.isnan(predicted_ratings[user_idx, item_idx]):
            predictions_dict[user_id][item_id] = predicted_ratings[user_idx, item_idx]
#%%
# save the rating to a file
# output_file_path_prediction = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\CF_ml\predicted_ratings_all_itemCF.pkl'
output_file_path_prediction = r'/home/msds/cli034/DataMiningproject/CF_ml/predicted_ratings_all_itemCF.pkl'
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
# output_file_path_recommendation = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\CF_ml\user_recommendation_itemCF.pkl'
output_file_path_recommendation = r'/home/msds/cli034/DataMiningproject/CF_ml/user_recommendation_userCF.pkl'
with open(output_file_path_recommendation, 'wb') as output_file_re:
    pickle.dump(recommendations_dict, output_file_re)

print("save successfully for recommendation")