import numpy as np
import random

random.seed(1234)

class DataInput:
	def __init__(self, data, u_his_list, i_his_list, batch_size, trunc_len, user_count, item_count):
		self.batch_size = batch_size
		self.data = data
		self.u_his_list = u_his_list
		self.i_his_list = i_his_list
		self.trunc_len = trunc_len
		self.user_count = user_count
		self.item_count = item_count
		self.epoch_size = len(self.data) // self.batch_size
		if self.epoch_size * self.batch_size < len(self.data):
			self.epoch_size += 1
		self.i = 0
	
	def __iter__(self):
		return self

	def __next__(self):
		if self.i == self.epoch_size:
			raise StopIteration

		ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
		self.i += 1

		uid_1, iid_1, label_1 = [], [], []
		uid_2, iid_2, label_2 = [], [], []
		u_his, u_his_l = [], []
		i_his, i_his_l = [], []

		for t in ts:
			uid_1.append(t[0])
			iid_1.append(t[1])
			label_1.append(t[2])
			u_his_all = self.u_his_list[t[0]]
			u_his_u = []
			for i in u_his_all:
				if i==t[1]:
					break
				u_his_u.append(i)
			u_his_u = u_his_u[max(len(u_his_u)-self.trunc_len, 0):len(u_his_u)]
			u_his_u_l = len(u_his_u)
			if len(u_his_u)<=0:
				u_his_u = [0]
			u_his.append(u_his_u)
			u_his_l.append(u_his_u_l)

			i_his_all = self.i_his_list[t[1]]
			i_his_i = []
			for u in i_his_all:
				if u==t[0]:
					break
				i_his_i.append(u)
			i_his_i = i_his_i[max(len(i_his_i)-self.trunc_len, 0):len(i_his_i)]
			i_his_i_l = len(i_his_i)
			if len(i_his_i)<=0:
				i_his_i = [0]
			i_his.append(i_his_i)
			i_his_l.append(i_his_i_l)

			#generate negative sample
			for i in range(5):
				uid_1.append(t[0])
				neg = np.random.randint(0, self.item_count)
				while neg==t[1]:
					neg = np.random.randint(0, self.item_count)
				iid_1.append(neg)
				label_1.append(0)
				u_his.append(u_his_u)
				u_his_l.append(u_his_u_l)
				i_his.append(i_his_i)
				i_his_l.append(i_his_i_l)

			#sample for user RNN
			uid_2.append(t[0])
			iid_2.append(t[1])
			label_2.append(t[2])
			for i in range(5):
				iid_2.append(t[1])
				neg = np.random.randint(0, self.user_count)
				while neg==t[0]:
					neg = np.random.randint(0, self.user_count)
				uid_2.append(neg)
				label_2.append(0)

		data_len = len(uid_1)
    
		#padding
		u_his_maxlength = max(max(u_his_l), 1)
		u_hisinput = np.zeros([data_len, u_his_maxlength], dtype = np.int32)
		for i, ru in enumerate(u_his):
			u_hisinput[i, :len(ru)] = ru[:len(ru)]
		i_his_maxlength = max(max(i_his_l), 1)
		i_hisinput = np.zeros([data_len, i_his_maxlength], dtype = np.int32)
		for i, ru in enumerate(i_his):
			i_hisinput[i, :len(ru)] = ru[:len(ru)]

		return self.i, (uid_1, iid_1, label_1), (uid_2, iid_2, label_2), u_hisinput, u_his_l, i_hisinput, i_his_l