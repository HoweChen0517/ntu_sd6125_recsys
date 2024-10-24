import pandas as pd
import pickle
import torch
#%%
# load the ratings.csv as original rating data
# ratings_file_path = r"E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mNew\ratings.csv"
ratings_file_path = r"/home/msds/cli034/DataMiningproject/ml-20mNew/ratings.csv"
ratings_df = pd.read_csv(ratings_file_path)

# load the predicted_ratings.pkl as predicted rating data
# predictions_file_path = r"E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\CF_ml\predicted_ratings_all_itemCF.pkl"
predictions_file_path = r"/home/msds/cli034/DataMiningproject/CF_ml/predicted_ratings_all_itemCF.pkl"
with open(predictions_file_path, 'rb') as file:
    predictions_dict = pickle.load(file)
#%%
# define the error function
def calculate_error(ratings_df, predictions_dict):
    # convert original rating to tensor
    actual_ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)

    # convert predicted rating to tensor
    predicted_ratings = torch.tensor([predictions_dict.get(user_id, {}).get(item_id, float('nan'))
                                      for user_id, item_id in zip(ratings_df['userId'], ratings_df['itemId'])],
                                      dtype=torch.float32)

    # compute the average error
    valid_mask = ~torch.isnan(predicted_ratings)
    valid_actual_ratings = actual_ratings[valid_mask]
    valid_predicted_ratings = predicted_ratings[valid_mask]

    if valid_actual_ratings.numel() == 0:
        return None
    average_error = torch.mean(torch.abs(valid_actual_ratings - valid_predicted_ratings)).item()
    return average_error
#%%
# compute the error for prediction
average_error = calculate_error(ratings_df, predictions_dict)
if average_error is not None:
    print(f"average error for item collaborative filtering is : {average_error}")
    # save the average error for prediction
    # with open(r"E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\CF_ml\average_error_itemCF.txt", 'w') as error_file:
    with open(r"/home/msds/cli034/DataMiningproject/CF_ml/average_error_itemCF.txt", 'w') as error_file:
        error_file.write(f"average error for item collaborative filtering is: {average_error}")
else:
    print("no rating for error computing")