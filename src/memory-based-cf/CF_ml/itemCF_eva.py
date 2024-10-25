import pandas as pd
import numpy as np
# %%
# Load the predicted ratings file
predicted_ratings_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\CF_mlTT\predicted_ratings_all_itemCF.csv'
predicted_ratings_df = pd.read_csv(predicted_ratings_file, index_col=0)
# %%
# Load the test ratings file
ratings_file = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\ml-20mTT\3.ml_20m_df_test.parquet'
ratings = pd.read_parquet(ratings_file)
#%%
print(ratings.shape)
# %%
# Iterate through each row in ratings to calculate the error
errors = []
squared_errors = []
for _, row in ratings.iterrows():
    user_id = int(row['userId'])
    item_id = str(int(row['itemId']))
    actual_rating = row['rating']

    if user_id in predicted_ratings_df.index and item_id in predicted_ratings_df.columns:
        predicted_rating = predicted_ratings_df.loc[user_id, item_id]
        error = abs(predicted_rating - actual_rating)
        squared_error = (predicted_rating - actual_rating) ** 2
        errors.append(error)
        squared_errors.append(squared_error)
#%%
# Calculate the mean absolute error (MAE)
if errors:
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(squared_errors))
    print(f'Mean Absolute Error (MAE) of the predicted ratings: {mae}')
    print(f'Root Mean Square Error (RMSE) of the predicted ratings: {rmse}')
    # Save the MAE/RMSE to a local file
    metrics_file_path = r'E:\AAASchoolLearning\StudyResource\GraduateNTU\ProjectSum\DataMiningproject\CF_mlTT\metrics_value_itemCF.txt'
    with open(metrics_file_path, 'w') as metrics_file:
        metrics_file.write(f'Mean Absolute Error (MAE): {mae}\n')
        metrics_file.write(f'Root Mean Square Error (RMSE): {rmse}\n')
    print(f'Metrics value saved to {metrics_file_path}')
else:
    print("No errors to calculate MAE and RMSE. The list of errors is empty.")