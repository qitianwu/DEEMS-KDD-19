import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import csv
import eval
from input import DataInput
from model import Model

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234) 

learning_rate = 10
keep_prob = 1
lambda_reg = 0.01
trunc_len = 20
train_batch_size = 64
test_batch_size = 64
h1 = 0.1
h2 = 0.1


workdir = '/cluster/home/it_stu11/qitian/KDD/data'
with open(workdir+'/Clothing_dataset_-2_new.pkl', 'rb') as f:
	train_set = pickle.load(f)
	test_set = pickle.load(f)
	u_his_list = pickle.load(f)
	i_his_list = pickle.load(f)
	user_count, item_count, example_count = pickle.load(f)
 
def calc_metric(score_label_u):
	score_label_u = sorted(score_label_u, key=lambda d:d[0], reverse=True)
	precision = eval.precision_k(score_label_u, 3)
	recall = eval.recall_k(score_label_u, 3)
	try:
		f1 = 2*precision*recall/(precision+recall)
	except:
		f1 = 0
	auc = eval.auc(score_label_u)
	ndcg = eval.ndcg_k(score_label_u, 3)
	return precision, recall, f1, auc, ndcg
		
def _eval(sess, model, test_set_list):
	loss_sum = 0.
	Precision = 0.
	Recall = 0.
	F1 = 0.
	AUC = 0.
	NDCG = 0.
	num = 0
	score_label_all = []
	for i in range(len(test_set_list)):
		uid = test_set_list[i][0][0]
		u_his_all = u_his_list[uid]
		test_set_list_u = test_set_list[i]
		uid1_list, iid1_list, label1_list = [], [], []
		uid2_list, iid2_list, label2_list = [], [], []
		u_his, u_pos, u_his_l = [], [], []
		i_his, i_pos, i_his_l = [], [], []
		for s in test_set_list_u:
			uid1_list.append(uid)
			iid1_list.append(s[1])
			label1_list.append(s[2])
			u_his_u, i_his_i = [], []
			u_pos_u, i_pos_i = [], []
			for i in u_his_all:
				if i==s[1]:
					break
				u_his_u.append(i)
			u_his_u = u_his_u[max(len(u_his_u)-trunc_len, 0):len(u_his_u)]
			for j in range(len(u_his_u)):
				u_pos_u.append(len(u_his_u)-j-1)
			u_his_l_u = len(u_his_u)
			if u_his_l_u<=0:
				u_his_u = [0]
			u_his.append(u_his_u)
			u_pos.append(u_pos_u)
			u_his_l.append(u_his_l_u)
			i_his_all = i_his_list[s[1]]
			for u in i_his_all:
				if u==s[0]:
					break
				i_his_i.append(u)
			i_his_i = i_his_i[max(len(i_his_i)-trunc_len, 0):len(i_his_i)]
			for j in range(len(i_his_i)):
				i_pos_i.append(len(i_his_i)-j-1)
			i_his_l_i = len(i_his_i)
			if i_his_l_i<=0:
				i_his_i = [0]
			i_his.append(i_his_i)
			i_pos.append(i_pos_i)
			i_his_l.append(i_his_l_i)
			for k in range(2):
				neg = s[1]
				while neg==s[1]:
					neg = np.random.randint(0, item_count)
				uid1_list.append(uid)
				iid1_list.append(neg)
				label1_list.append(0)
				u_his.append(u_his_u)
				u_pos.append(u_pos_u)
				u_his_l.append(u_his_l_u)
				i_his.append(i_his_i)
				i_pos.append(i_pos_i)
				i_his_l.append(i_his_l_i)
			uid2_list.append(uid)
			iid2_list.append(s[1])
			label2_list.append(s[2])
			for k in range(2):
				neg = s[0]
				while neg==s[0]:
					neg = np.random.randint(0, user_count)
				uid2_list.append(neg)
				iid2_list.append(s[1])
				label2_list.append(0)
		u_his_maxlength = max(max(u_his_l), 1)
		u_hisinput = np.zeros([len(uid1_list), u_his_maxlength], dtype=np.int32)
		for i, ru in enumerate(u_his):
			u_hisinput[i, :len(ru)] = ru[:len(ru)]
		u_posinput = np.zeros([len(uid1_list), u_his_maxlength], dtype = np.int32)
		for i, ru in enumerate(u_pos):
			u_posinput[i, :len(ru)] = ru[:len(ru)]
		i_his_maxlength = max(max(i_his_l), 1)
		i_hisinput = np.zeros([len(iid1_list), i_his_maxlength], dtype=np.int32)
		for i, ru in enumerate(i_his):
			i_hisinput[i, :len(ru)] = ru[:len(ru)]
		i_posinput = np.zeros([len(iid1_list), i_his_maxlength], dtype = np.int32)
		for i, ru in enumerate(i_pos):
			i_posinput[i, :len(ru)] = ru[:len(ru)]
		datainput1 = (uid1_list, iid1_list, label1_list)
		datainput2 = (uid2_list, iid2_list, label2_list)
		score, loss = model.eval(sess, datainput1, datainput2, u_hisinput, u_posinput, u_his_l, i_hisinput, i_posinput, i_his_l)
		score_label_u = []
		for i in range(len(score)):
			score_label_u.append([score[i], label1_list[i]])
			score_label_all.append([score[i], label1_list[i]])
		precision, recall, f1, auc, ndcg = calc_metric(score_label_u)
		loss_sum += loss
		Precision += precision
		Recall += recall
		F1 += f1
		AUC += auc
		NDCG += ndcg
		num += 1
	score_label_all = sorted(score_label_all, key=lambda d:d[0], reverse=True)
	GP = eval.precision_k(score_label_all, 0.3*len(score_label_all))
	GAUC = eval.auc(score_label_all)
	return loss_sum/num, Precision/num, Recall/num, F1/num, AUC/num, NDCG/num, GP, GAUC

#log_txt = open('/home/myronwu/ijcai2019/code/MetaDSR/log/'+'2018-12-24-1.txt', 'w')

test_set_df = pd.DataFrame(test_set, columns=['uid', 'iid', 'label'])
test_set_list = []
for uid, hist in test_set_df.groupby('uid'):
	test_set_list_u = []
	for i in range(hist.shape[0]):
		test_set_list_u.append([hist.iloc[i][0], hist.iloc[i][1], hist.iloc[i][2]])
	test_set_list.append(test_set_list_u)
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session() as sess:
	model = Model(user_count, item_count)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer()) 
	#model.restore(sess, '/home/myronwu/save_model/DUAL_GAT.ckpt')

	sys.stdout.flush()
	start_time = time.time()
	Train_loss_pre = 100
	bestP, bestR, bestF1, bestAUC = 0.0, 0.0, 0.0, 0.0
	for _ in range(10000):

		random.shuffle(train_set)
		epoch_size = round(len(train_set) / train_batch_size)
		iter_num, loss_r_sum, loss_reg_sum = 0, 0., 0.
		for _, datainput1, datainput2, u_hisinput, u_posinput, u_his_l, i_hisinput, i_posinput, i_his_l in DataInput(train_set, u_his_list, i_his_list, train_batch_size, trunc_len, user_count, item_count):
			loss_r, loss_reg = model.train(sess, datainput1, datainput2, u_hisinput, u_posinput, u_his_l, i_hisinput, i_posinput, i_his_l, learning_rate, keep_prob, lambda_reg, h1, h2)
			iter_num += 1
			loss_r_sum += loss_r
			loss_reg_sum += loss_reg

			if model.global_step.eval() % 1000 == 0:
				Train_loss_r = loss_r_sum / iter_num
				Train_loss_reg = loss_reg_sum / iter_num
				Test_loss, P, R, F1, AUC, NDCG, GP, GAUC = _eval(sess, model, test_set_list)
				print('Epoch %d Step %d Train: %.4f Reg: %.4f Test: %.4f P: %.4f R: %.4f F1: %.4f AUC: %.4f' %
				(model.global_epoch_step.eval(), model.global_step.eval(), Train_loss_r, Train_loss_reg, Test_loss, P, R, F1, AUC))
				print('Best P: %.4f Best R: %.4f Best F1: %.4f Best AUC: %.4f' %
				(bestP, bestR, bestF1, bestAUC))
				iter_num = 0
				loss_r_sum, loss_reg_sum = 0., 0.
				if AUC > bestAUC:
					model.save(sess, '/cluster/home/it_stu11/qitian/KDD/save_model/DEEMS_ATTN.ckpt') 
					bestAUC = AUC
				if P > bestP: 
					bestP = P
				if R > bestR: 
					bestR = R
				if F1 > bestF1: 
					bestF1 = F1
				#log_txt.write('Epoch %d Step %d Train_r_loss: %.4f Train_loss_sn: %.4f Test_loss: %.4f MAE: %.4f MRSE: %.4f Best_mae: %.4f Best_RMSE: %.4f' %
				#(model.global_epoch_step.eval(), model.global_step.eval(), Train_loss_r, Train_loss_sn, Test_loss, MAE, RMSE, best_mae, best_rmse))

		#print('Epoch %d DONE\tCost time: %.2f' %
		#(model.global_epoch_step.eval(), time.time()-start_time))
		sys.stdout.flush()
		model.global_epoch_step_op.eval()
		if model.global_epoch_step.eval() % 10 == 9:
			h1 = h1*0.1
			h2 = h2*0.1
			learning_rate = learning_rate*0.3
		#	learning_rate_sn = learning_rate_sn*0.9

		#if abs(Train_loss-Train_loss_pre) < 1e-6:
		#	break
		#Train_loss_pre = Train_loss
	

	sys.stdout.flush()
	
