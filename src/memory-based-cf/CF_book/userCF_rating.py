#%%
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
#%%
# load the user_sim data file
userCF_sim_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\book_crossingTT\5.book_crossing_user_sim.parquet'
user_sim = pd.read_parquet(userCF_sim_file)
#%%
# retrieve all the userId & movieId from the training set
ratings_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\book_crossingTT\2.book_crossing_df_train.parquet'
ratings = pd.read_parquet(ratings_file)
ratings = ratings[ratings['userId'].isin(user_sim.index)]

all_users = ratings['userId'].unique()

#%%
# Re-map
# Re-map user IDs to ensure consistent indexing
unique_users = sorted(ratings['userId'].unique())
user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
# Create reverse mapping to convert indices back to original user IDs
reverse_user_mapping = {idx: user_id for user_id, idx in user_mapping.items()}

# Re-map itemId to make coo-matrix in order and in shape
unique_items = sorted(ratings['itemId'].unique())
item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
# Create reverse mapping to convert indices back to original item IDs
reverse_item_mapping = {idx: item_id for item_id, idx in item_mapping.items()}


ratings['userId'] = ratings['userId'].map(user_mapping)

#%%
# Update user similarity matrix to use new index
user_sim = user_sim.loc[unique_users, unique_users]

#%%
# Convert updated user_sim to a sparse matrix with correct dimensions
user_sim_values = user_sim.stack().values
user_sim_rows = user_sim.stack().index.get_level_values(0).map(user_mapping)
user_sim_cols = user_sim.stack().index.get_level_values(1).map(user_mapping)
# Ensure the indices are within the correct range
user_sim_matrix = coo_matrix(
    (user_sim_values, (user_sim_rows, user_sim_cols)),
    shape=(len(unique_users), len(unique_users))
).tocsr()
#%%
# Update user_item_matrix to use new item indices
ratings['userId'] = ratings['userId'].map(user_mapping)
ratings['itemId'] = ratings['itemId'].map(item_mapping)
user_item_matrix = coo_matrix(
    (ratings['rating'], (ratings['userId'], ratings['itemId'])))

#%%
# transpose the user_item_matrix to match dimension
user_item_matrix_transposed = user_item_matrix.T
#%%
# the function to predict user's rating for all movies
def predict_ratings_matrix(user_item_matrix_transposed, user_sim_matrix):
    # Predict ratings by multiplying user-item matrix with item similarity matrix
    predicted_ratings = user_item_matrix_transposed.dot(user_sim_matrix)
    return predicted_ratings
#%%
# # Calculate all users' predicted ratings for all items in a batch
predicted_ratings = predict_ratings_matrix(user_item_matrix_transposed, user_sim_matrix)
#%%
# transpose the predicted_ratings to make row index as user_id and column as item_id
predicted_ratings_transposed = predicted_ratings.T
#%%
# reverse the original itemId & userId
predicted_ratings_df = pd.DataFrame(predicted_ratings_transposed.toarray())
predicted_ratings_df.index = predicted_ratings_df.index.map(reverse_user_mapping)
predicted_ratings_df.columns = predicted_ratings_df.columns.map(reverse_item_mapping)
#%%
# Calculate row sums of user_sim_matrix and divide predicted_ratings_df rows by these sums
row_sums = user_sim_matrix.sum(axis=1).A1  # Convert to a 1D array
predicted_ratings_normalized_df = predicted_ratings_df.div(row_sums, axis=0)
#%%
# save the prediction rating
output_file_path_complete_prediction = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\CF_bookTT\predicted_ratings_all_userCF.csv'
predicted_ratings_normalized_df.to_csv(output_file_path_complete_prediction)
print("save successfully for complete rating matrix with original user IDs")
#%%
# Mask the original ratings to keep only predicted ratings for unscored items
mask_normalized = user_item_matrix > 0
predicted_ratings_normalized_df[mask_normalized.toarray()] = 0
#%%
# create a recommendation dict for user, sorted by rating
recommendations_dict = {}

for user_idx, user_id in enumerate(all_users):
    user_ratings = predicted_ratings_normalized_df.loc[user_id]
    sorted_indices = user_ratings.sort_values(ascending=False).index  # Sort in descending order
    top_items = sorted_indices[:100].tolist()  # Select top 100 items
    recommendations_dict[user_id] = top_items
#%%
# Save the recommendations dict to a file
output_file_path_recommendation = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\CF_bookTT\user_recommendation_userCF.pkl'
with open(output_file_path_recommendation, 'wb') as output_file_re:
    pickle.dump(recommendations_dict, output_file_re)

print("save successfully for recommendation")
