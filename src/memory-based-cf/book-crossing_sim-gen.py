import cf.preprocessing.load_data as load_data
import cf.preprocessing.inputs as inputs

data_set = 'book_crossing'
data = load_data.read_data(data_set=data_set)
splitter = load_data.train_test_split(data=data, test_size=0.2, random_state=42)
trn_data, val_data = splitter.train_test_split_random(data, test_size=0.2, random_state=42)
transformer = load_data.data_transformation(trn_data)
user_index, item_index, rating_matrix = transformer.convert_df_to_sparse_matrix(trn_data, flag='score')
generator = inputs.CFinputs(rating_matrix, user_index, item_index, data_set, batch_size=1000)
generator.get_itemCF_sim_batch()
generator = inputs.CFinputs(rating_matrix, user_index, item_index, data_set, batch_size=1000)
generator.get_userCF_sim_batch()