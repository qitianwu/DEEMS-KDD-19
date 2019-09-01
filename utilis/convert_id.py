import csv
import pandas as pd
import numpy as np
import pickle

def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key

workdir = '/home/wuiron/data'

csv_reader = csv.reader(open(workdir+'/Baby.csv', 'r'))
#csv_reader = pd.read_csv(workdir+'/Baby.csv', encoding='ANSI')

rate_list = [] #selected users and items
rate_list_ = [] #all users and items
for s in csv_reader:
  uid = s[0]
  iid = s[1]
  rate = s[2]
  try:
    time = int(s[3])
  except:
    continue
  rate_list_.append([uid, iid, rate, time])
rate_df_ = pd.DataFrame(rate_list_, columns=['uid', 'iid', 'rate', 'time'])
mintime = rate_df_['time'].min()
print(rate_df_.head())

uid_map, uid_key = build_map(rate_df_, 'uid')
iid_map, iid_key = build_map(rate_df_, 'iid')
print(rate_df_.head())
user_count_, item_count_, example_count_ =\
    len(uid_map), len(iid_map), rate_df_.shape[0]
print('Raw Statistics: user_count: %d\titem_count: %d\texample_count: %d' %
      (user_count_, item_count_, example_count_))

rate_list_ = []
for i in range(example_count_):
  s = rate_df_.iloc[i].tolist()
  rate_list_.append(s)

item_num = np.zeros(user_count_)
user_num = np.zeros(item_count_)
for s in rate_list_:
  uid = s[0]
  iid = s[1]
  item_num[uid] = item_num[uid] + 1
  user_num[iid] = user_num[iid] + 1
for s in rate_list_:
  uid = s[0]
  iid = s[1]
  rate = s[2]
  time = s[3] - mintime
  if item_num[uid]>=3 and user_num[iid]>=3:
    rate_list.append([uid, iid, rate, time])
rate_df = pd.DataFrame(rate_list, columns = ['uid', 'iid', 'rate', 'time'])

uid_map, uid_key = build_map(rate_df, 'uid')
iid_map, iid_key = build_map(rate_df, 'iid')
print(rate_df.head())
user_count, item_count, example_count =\
    len(uid_map), len(iid_map), rate_df.shape[0]
print('New Statistics: user_count: %d\titem_count: %d\texample_count: %d' %
      (user_count, item_count, example_count))
'''
with open(workdir + '/Digital_remap_raw.pkl', 'wb') as f:
  pickle.dump(rate_df_, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count_, item_count_, example_count_), f, pickle.HIGHEST_PROTOCOL)
'''
with open(workdir + '/Baby_remap_3.pkl', 'wb') as f:
  pickle.dump(rate_df, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, example_count), f, pickle.HIGHEST_PROTOCOL)
