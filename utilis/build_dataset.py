import random
import pickle
import numpy as np
import pandas as pd

random.seed(1234)

workdir = '/cluster/home/it_stu11/qitian/KDD/data'
with open(workdir+'/Clothing_remap.pkl', 'rb') as f:
	rate_df = pickle.load(f)
	user_count, item_count, example_count = pickle.load(f)

u_his_list = [] #[U, I]
i_his_list = [] #[I, U]
pos_list = []

train_set, test_set = [], []
for uid, hist in rate_df.groupby('uid'):
	u_hist_df = hist.sort_index(by='time', axis=0, ascending=True)
	u_his_ui = u_hist_df['iid'].tolist()
	u_his_list.append(u_his_ui)
	
	pos_list_u, neg_list_u = [], []
	for iid in u_his_ui:
		pos_list_u.append([uid, iid, 1])
	random.shuffle(pos_list_u)
	if len(pos_list_u)>5:
		train_set = train_set + pos_list_u[:-2]
		test_set = test_set + pos_list_u[-2:len(pos_list_u)]
	else:
		train_set = train_set + pos_list_u
	
	'''
	def gen_neg_i(pos_list):
		neg = pos_list[0]
		while neg in pos_list:
			neg = np.random.randint(0, item_count)
		return neg
	for i in range(len(u_his_ui)):
		iid = gen_neg_i(u_his_ui)
		neg_list_u.append([uid, iid, 0])
	'''
	#train_set = train_set + pos_list_u[:-2]
	#test_set = test_set + pos_list_u[-2:len(pos_list_u)]

#train_set_i, test_set_i = [], []
for iid, hist in rate_df.groupby('iid'):
	i_hist_df = hist.sort_index(by='time', axis=0, ascending=True)
	i_his_iu = i_hist_df['uid'].tolist()
	i_his_list.append(i_his_iu)
	'''
	pos_list_i, neg_list_i = [], []
	for uid in i_his_iu:
		pos_list_i.append([uid, iid, 1])
	random.shuffle(pos_list_i)
	train_set = train_set + pos_list_i[:-2]
	test_set = test_set + pos_list_i[-2:len(pos_list_i)]
	'''
'''
	pos_list_i, neg_list_i = [], []
	for uid in i_his_iu:
		pos_list_i.append([uid, iid, 1])
	def gen_neg_u(pos_list):
		neg = pos_list[0]
		while neg in pos_list:
			neg = np.random.randint(0, user_count)
		return neg
	for i in range(len(i_his_iu)):
		uid = gen_neg_u(i_his_iu)
		neg_list_i.append([uid, iid, 0])
	train_set_i = train_set_i + pos_list_i[:-2] + neg_list_i[:-2]
	test_set_i = test_set_i + pos_list_i[-2:len(pos_list_i)] + neg_list_i[-2:len(neg_list_i)]



random.shuffle(train_set_i)
random.shuffle(test_set_i)
print('For items, Train size: ', len(train_set_i), 'Test size: ', len(test_set_i))
print(train_set_i[:5])
print(test_set_i[:5])
'''

'''
train_set, test_set = [], []
rate_df_time = rate_df.sort_index(by='time', axis=0, ascending=True)
for i in range(int(0.8*(rate_df_time.shape[0]))):
	train_set.append([rate_df_time.iloc[i][0], rate_df_time.iloc[i][1], 1])
for i in range(int(0.8*(rate_df_time.shape[0])), rate_df_time.shape[0]):
	test_set.append([rate_df_time.iloc[i][0], rate_df_time.iloc[i][1], 1])
'''

random.shuffle(train_set)
random.shuffle(test_set)
print('For users, Train size: ', len(train_set), 'Test size: ', len(test_set))
print(train_set[:5])
print(test_set[:5])
 
with open(workdir+'/Clothing_dataset_-2_new.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
	#pickle.dump(train_set_i, f, pickle.HIGHEST_PROTOCOL)
	#pickle.dump(test_set_i, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_his_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_his_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((user_count, item_count, example_count), f, pickle.HIGHEST_PROTOCOL)


