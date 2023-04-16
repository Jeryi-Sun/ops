#coding=utf-8


"""data files info"""

root_path = '/Users/sunzhongxiang/Desktop/科研/因搜索打开app/dataset/raw_dataset/new_dataset/'
#root_path = '../data/'
ckpt = 'ckpt'


train_file = root_path+'train.tsv'
valid_file = root_path+'valid.tsv'
test_file = root_path+'test.tsv'
open_search_index = root_path+"open_search_index.pickle"
search_index = root_path+"search_index.pickle"
recommendation_index = root_path+"recommendation_index.pickle"

search_vocab = root_path+"search_vocab.pickle"
reco_vocab = root_path+"reco_vocab.pickle"
user_vocab = root_path+"user_vocab.pickle"
open_actions = root_path+'open_action.pickle'

"""item/user/query feature"""

item_id_num = 936684 + 1 #zero for padding
item_id_dim = 26 
item_type1_num = 38
item_type1_dim = 8
item_cate_num = 37
item_cate_dim = 8


user_id_num = 1915
user_id_dim = 48
user_gender_num = 3
user_gender_dim = 4
user_age_num = 8
user_age_dim = 4
user_src_level_num = 4
user_src_level_dim = 4

query_id_num = 33085 + 1 #zero for padding
query_id_dim = 26
query_search_source_num = 4
query_search_source_dim = 48
user_search_num = 116569 + 1 #zero for padding
user_search_dim = 48

time_feature_dim=6
"""experiment config"""
# max_rec_his_len = 150 
# max_words_of_item = 20 # maximum number of words in subtile and caption of an item 
# max_words_of_query = 6 # maximum number of words in a query 
# max_src_his_len = 25
# max_src_click_item = 5 # maximum number of clicked items for a query in one session
# n_layers = 2
max_seq_len_reco = 20
max_seq_len_search = 10
max_seq_len_open_search = 5