import numpy as np
import random

random.seed(1234)

class DataInput:
	def __init__(self, data, u_his_list, batch_size, trunc_len, item_count):
		self.batch_size = batch_size
		self.data = data
		self.u_his_list = u_his_list
		self.trunc_len = trunc_len
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

		uid, iid, label = [], [], []
		u_his, u_his_l = [], []
		u_pos = []

		for t in ts:
			uid.append(t[0])
			iid.append(t[1])
			label.append(t[2])
			u_his_all = self.u_his_list[t[0]]
			u_his_u, u_pos_u = [], []
			for i in u_his_all:
				if i==t[1]:
					break
				u_his_u.append(i)
			u_his_u = u_his_u[max(len(u_his_u)-self.trunc_len, 0):len(u_his_u)]
			for j in range(len(u_his_u)):
				u_pos_u.append(len(u_his_u)-j-1)
			u_his_u_l = len(u_his_u)
			if len(u_his_u)<=0:
				u_his_u = [0]
				u_pos_u = [0]
			u_his.append(u_his_u)
			u_pos.append(u_pos_u)
			u_his_l.append(u_his_u_l)

			#generate negative sample
			for i in range(5):
				uid.append(t[0])
				neg = np.random.randint(0, self.item_count)
				while neg==t[1]:
					neg = np.random.randint(0, self.item_count)
				iid.append(neg)
				label.append(0)
				u_his.append(u_his_u)
				u_pos.append(u_pos_u)
				u_his_l.append(u_his_u_l)

		data_len = len(uid)
    
		#padding
		u_his_maxlength = max(max(u_his_l), 1)
		u_hisinput = np.zeros([data_len, u_his_maxlength], dtype = np.int32)
		for i, ru in enumerate(u_his):
			u_hisinput[i, :len(ru)] = ru[:len(ru)]
		u_posinput = np.zeros([data_len, u_his_maxlength], dtype = np.int32)
		for i, ru in enumerate(u_pos):
			u_posinput[i, :len(ru)] = ru[:len(ru)]

		return self.i, (uid, iid, label), u_hisinput, u_posinput, u_his_l