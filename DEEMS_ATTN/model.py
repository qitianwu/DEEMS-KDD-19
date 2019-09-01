import tensorflow as tf

class Model(object):
    
	def __init__(self, user_count, item_count):
 
		self.user1 = tf.placeholder(tf.int32, [None,]) # [B]
		self.item1 = tf.placeholder(tf.int32, [None,]) # [B]
		self.label1 = tf.placeholder(tf.float32, [None,]) # [B]
		self.user2 = tf.placeholder(tf.int32, [None,]) # [B]
		self.item2 = tf.placeholder(tf.int32, [None,]) # [B]
		self.label2 = tf.placeholder(tf.float32, [None,]) # [B]
		self.u_his = tf.placeholder(tf.int32, [None, None]) # [B, I]
		self.u_pos = tf.placeholder(tf.int32, [None, None]) # [B, I]
		self.u_his_l = tf.placeholder(tf.float32, [None,]) # [B]
		self.i_his = tf.placeholder(tf.int32, [None, None]) # [B, U]
		self.i_pos = tf.placeholder(tf.int32, [None, None]) # [B, U]
		self.i_his_l = tf.placeholder(tf.float32, [None,]) # [B]

		self.lr = tf.placeholder(tf.float32)
		self.reg = tf.placeholder(tf.float32) 
		self.isTrain = tf.placeholder(tf.int32)
		self.kp = tf.placeholder(tf.float32)
		self.hedge1 = tf.placeholder(tf.float32)
		self.hedge2 = tf.placeholder(tf.float32)

		hidden_units_u = 20
		hidden_units_i = 20

		def sequence_mask(object_, key):
			key_masks = tf.sequence_mask(key, tf.shape(object_)[1])
			key_masks = tf.expand_dims(key_masks, axis = 2)
			key_masks = tf.tile(key_masks, [1, 1, tf.shape(object_)[2]])
			key_masks = tf.reshape(key_masks, [-1, tf.shape(object_)[1], tf.shape(object_)[2]])
			paddings = tf.zeros_like(object_)
			output = tf.where(key_masks, object_, paddings)
			return output

		def RNN(x, hidden_num, name):
			x = tf.reshape(x, shape=[-1, tf.shape(x)[1], hidden_num])
			rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num, name=name, reuse=tf.AUTO_REUSE)
			output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
			return states
		
		def attention(query, key, name):
			query_ = tf.expand_dims(query, axis=1)
			query_ = tf.tile(query_, [1, tf.shape(key)[1], 1])
			attn_in = tf.concat([query_, key], axis=2)
			weights = tf.layers.dense(attn_in, 1, activation=tf.nn.tanh, use_bias = True, name=name, reuse=tf.AUTO_REUSE)
			weights = tf.nn.softmax(weights, axis=1)
			weights_ = tf.tile(weights, [1, 1, tf.shape(key)[2]])
			output = tf.reduce_sum(tf.multiply(weights_, key), axis=1)
			return output

		user1_emb = tf.get_variable("u_user_emb", [user_count, hidden_units_u])
		item1_emb = tf.get_variable("u_item_emb", [item_count, hidden_units_i])
		user1_b = tf.get_variable("u_user_b", [user_count],
                             initializer=tf.constant_initializer(0.0))
		item1_b = tf.get_variable("u_item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
		u_pos_emb = tf.get_variable("u_pos_emb", [20, hidden_units_i])
		user2_emb = tf.get_variable("i_user_emb", [user_count, hidden_units_u])
		item2_emb = tf.get_variable("i_item_emb", [item_count, hidden_units_i])
		user2_b = tf.get_variable("i_user_b", [user_count],
                             initializer=tf.constant_initializer(0.0))
		item2_b = tf.get_variable("i_item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
		i_pos_emb = tf.get_variable("i_pos_emb", [20, hidden_units_u])

		#--------------user-aspect RNN model-------------------
         
		u1s_emb = tf.nn.embedding_lookup(user1_emb, self.user1)
		i1s_emb = tf.nn.embedding_lookup(item1_emb, self.item1)
		u1s_b = tf.gather(user1_b, self.user1)
		i1s_b = tf.gather(item1_b, self.item1)
		u1r_emb = tf.nn.embedding_lookup(user2_emb, self.user1)
		i1r_emb = tf.nn.embedding_lookup(item2_emb, self.item1)
		u1r_b = tf.gather(user2_b, self.user1)
		i1r_b = tf.gather(item2_b, self.item1)
		u_hiss_emb = tf.nn.embedding_lookup(item1_emb , self.u_his)
		i_hisr_emb = tf.nn.embedding_lookup(user2_emb, self.i_his)
		u_hiss_emb = sequence_mask(u_hiss_emb, self.u_his_l)
		i_hisr_emb = sequence_mask(i_hisr_emb, self.i_his_l)
		u_poss_emb = tf.nn.embedding_lookup(u_pos_emb , self.u_pos)
		i_posr_emb = tf.nn.embedding_lookup(i_pos_emb, self.i_pos)
		u_poss_emb = sequence_mask(u_poss_emb, self.u_his_l)
		i_posr_emb = sequence_mask(i_posr_emb, self.i_his_l)	
		
		u_ds_emb = attention(i1s_emb, u_hiss_emb+u_poss_emb, 'u_att_l1')
		u_ds_emb = tf.reshape(u_ds_emb, [-1, hidden_units_i])
		i_dr_emb = attention(u1r_emb, i_hisr_emb+i_posr_emb, 'i_att_l1')
		i_dr_emb = tf.reshape(i_dr_emb, [-1, hidden_units_u])
		
		#u_nn_ins = tf.concat([u_ds_emb, i1s_emb, u1s_emb], axis=-1)
		#u_nn_l1s = tf.layers.dense(u_ds_emb, 1, activation=None, use_bias = True, name='u_nn_l1', reuse=tf.AUTO_REUSE)
		#u_nn_l1s = tf.reshape(u_nn_l1s, [-1])
		#i_nn_inr = tf.concat([i_dr_emb, i1r_emb, u1r_emb], axis=-1)
		#i_nn_l1r = tf.layers.dense(i_dr_emb, 1, activation=None, use_bias = True, name='i_nn_l1', reuse=tf.AUTO_REUSE)
		#i_nn_l1r = tf.reshape(i_nn_l1r, [-1])
		self.logits_us = u1s_b + i1s_b + tf.reduce_sum(tf.multiply(u_ds_emb + i1s_emb, u1s_emb), axis=1)
		self.logits_ir = u1r_b + i1r_b + tf.reduce_sum(tf.multiply(i_dr_emb + i1r_emb, u1r_emb), axis=1)
		self.pred_us = tf.nn.sigmoid(self.logits_us)
		self.pred_ir = tf.nn.sigmoid(self.logits_ir)

		#--------------item-aspect RNN model-------------------


		u2s_emb = tf.nn.embedding_lookup(user2_emb, self.user2)
		i2s_emb = tf.nn.embedding_lookup(item2_emb, self.item2)
		u2s_b = tf.gather(user2_b, self.user2)
		i2s_b = tf.gather(item2_b, self.item2)
		u2r_emb = tf.nn.embedding_lookup(user1_emb, self.user2)
		i2r_emb = tf.nn.embedding_lookup(item1_emb, self.item2)
		u2r_b = tf.gather(user1_b, self.user2)
		i2r_b = tf.gather(item1_b, self.item2)
		u_hisr_emb = tf.nn.embedding_lookup(item2_emb , self.u_his)
		i_hiss_emb = tf.nn.embedding_lookup(user2_emb, self.i_his)
		u_hisr_emb = sequence_mask(u_hisr_emb, self.u_his_l)
		i_hiss_emb = sequence_mask(i_hiss_emb, self.i_his_l)
		u_posr_emb = tf.nn.embedding_lookup(u_pos_emb , self.u_pos)
		i_poss_emb = tf.nn.embedding_lookup(i_pos_emb, self.i_pos)
		u_posr_emb = sequence_mask(u_posr_emb, self.u_his_l)
		i_poss_emb = sequence_mask(i_poss_emb, self.i_his_l)

		u_dr_emb = attention(i1r_emb, u_hisr_emb+u_posr_emb, 'u_att_l1')
		u_dr_emb = tf.reshape(u_dr_emb, [-1, hidden_units_i])
		i_ds_emb = attention(u1s_emb, i_hiss_emb+i_poss_emb, 'i_att_l1')
		i_ds_emb = tf.reshape(i_ds_emb, [-1, hidden_units_u])

		#u_nn_inr = tf.concat([u_dr_emb, i2r_emb, u2r_emb], axis=-1)
		#u_nn_l1r = tf.layers.dense(u_dr_emb, 1, activation=None, use_bias = True, name='u_nn_l1', reuse=tf.AUTO_REUSE)
		#u_nn_l1r = tf.reshape(u_nn_l1r, [-1])
		#i_nn_ins = tf.concat([i_ds_emb, i2s_emb, u2s_emb], axis=-1)
		#i_nn_l1s = tf.layers.dense(i_ds_emb, 1, activation=None, use_bias = True, name='i_nn_l1', reuse=tf.AUTO_REUSE)
		#i_nn_l1s = tf.reshape(i_nn_l1s, [-1])
		self.logits_ur = u2r_b + i2r_b + tf.reduce_sum(tf.multiply(u_dr_emb + i2r_emb, u2r_emb), axis=1)
		self.logits_is = u2s_b + i2s_b + tf.reduce_sum(tf.multiply(i_ds_emb+ i2s_emb, u2s_emb), axis=1)
		self.pred_ur = tf.nn.sigmoid(self.logits_ur)
		self.pred_is = tf.nn.sigmoid(self.logits_is)

		#--------------loss computation-------------------

		self.logits = 0.5*(self.logits_us + self.logits_is)
		self.logits = tf.reshape(self.logits, [-1])
		self.loss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label1, logits=self.logits))
		
		self.loss_r1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label1, logits=self.logits_us))
		#self.reward1 = tf.nn.sigmoid(tf.abs(self.pred_us-self.pred_ir))
		#self.reward1 = tf.math.exp(-tf.square(self.pred_us-self.pred_ir))
		#self.reward1 = tf.log(self.pred_us+0.001) - tf.log(self.pred_ir+0.001)
		#self.reward1 = tf.nn.sigmoid(tf.math.maximum(self.pred_us-self.pred_ir-0.2, 0))
		self.reward1 = -tf.reduce_mean(tf.multiply(self.pred_ir, tf.math.log(self.pred_us+1e-6))+tf.multiply(1-self.pred_ir, tf.math.log(1-self.pred_us+1e-6)))
		#self.loss_hedge1 = tf.reduce_mean(tf.multiply(1-self.label1, tf.multiply(tf.log(self.pred_us+0.001), self.reward1)))
		self.loss_hedge1 = tf.reduce_mean(tf.multiply(1-self.label1, self.reward1))
		self.loss_reg1 = tf.reduce_sum(tf.square(u1s_emb)) + tf.reduce_sum(tf.square(i1s_emb)) + tf.reduce_sum(tf.square(u_hiss_emb))
		self.loss1 = self.loss_r1 + self.reg*self.loss_reg1 + self.hedge1*self.loss_hedge1 
		self.loss_r2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label2, logits=self.logits_is))
		#self.reward2 = tf.nn.sigmoid(tf.abs(self.pred_is-self.pred_ur))
		#self.reward2 = tf.math.exp(-tf.square(self.pred_is-self.pred_ur))
		#self.reward2 = tf.log(self.pred_is+0.001) - tf.log(self.pred_ur+0.0001)
		#self.reward2 = tf.nn.sigmoid(tf.math.maximum(self.pred_is-self.pred_ur-0.2, 0))
		self.reward2 = -tf.reduce_mean(tf.multiply(self.pred_is, tf.math.log(self.pred_ur+1e-6))+tf.multiply(1-self.pred_is, tf.math.log(1-self.pred_ur+1e-6)))
		#self.loss_hedge2 = tf.reduce_mean(tf.multiply(1-self.label2, tf.multiply(tf.log(self.pred_is+0.001), self.reward2)))
		self.loss_hedge2 = tf.reduce_mean(tf.multiply(1-self.label2, self.reward2))
		self.loss_reg2 = tf.reduce_sum(tf.square(u2s_emb)) + tf.reduce_sum(tf.square(i2s_emb)) + tf.reduce_sum(tf.square(i_hiss_emb))
		self.loss2 = self.loss_r2 + self.reg*self.loss_reg2 + self.hedge2*self.loss_hedge2  
		

		# Step variable
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.global_epoch_step = \
		tf.Variable(0, trainable=False, name='global_epoch_step')
		self.global_epoch_step_op = \
		tf.assign(self.global_epoch_step, self.global_epoch_step+1)

		self.opt1 = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		#self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
		trainable_params1 = tf.trainable_variables('u')
		gradients1 = tf.gradients(self.loss1, trainable_params1)
		clip_gradients1, _ = tf.clip_by_global_norm(gradients1, 20*self.lr)
		self.train_op1 = self.opt1.apply_gradients(zip(clip_gradients1, trainable_params1), global_step=self.global_step)

		self.opt2 = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		#self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
		trainable_params2 = tf.trainable_variables('i')
		gradients2 = tf.gradients(self.loss2, trainable_params2)
		clip_gradients2, _ = tf.clip_by_global_norm(gradients2, 20*self.lr)
		self.train_op2 = self.opt2.apply_gradients(zip(clip_gradients2, trainable_params2))
		
		#--------------end model---------------

	def train(self, sess, datainput1, datainput2, u_hisinput, u_posinput, u_his_l, i_hisinput, i_posinput, i_his_l, lr, kp, reg, h1, h2):
		loss_r, loss_reg, _ = sess.run([self.loss_r, self.loss_reg1, self.train_op1], feed_dict={
		self.user1: datainput1[0], self.item1: datainput1[1], self.label1: datainput1[2], \
		self.user2: datainput2[0], self.item2: datainput2[1], self.label2: datainput2[2], \
		self.u_his: u_hisinput, self.u_pos: u_posinput, self.u_his_l: u_his_l, self.i_his: i_hisinput, self.i_pos: i_posinput, self.i_his_l: i_his_l, \
		self.lr: lr, self.reg: reg, self.isTrain: 1, self.kp: kp, self.hedge1: h1, self.hedge2: h2,
		})
		loss_r, _ = sess.run([self.loss_r, self.train_op2], feed_dict={
		self.user1: datainput1[0], self.item1: datainput1[1], self.label1: datainput1[2], \
		self.user2: datainput2[0], self.item2: datainput2[1], self.label2: datainput2[2], \
		self.u_his: u_hisinput, self.u_pos: u_posinput, self.u_his_l: u_his_l, self.i_his: i_hisinput, self.i_pos: i_posinput, self.i_his_l: i_his_l, \
		self.lr: lr, self.reg: reg, self.isTrain: 1, self.kp: kp, self.hedge1: h1, self.hedge2: h2,
		})
		return loss_r, loss_reg

	def eval(self, sess, datainput1, datainput2, u_hisinput, u_posinput, u_his_l, i_hisinput, i_posinput, i_his_l):
		logits, loss_r = sess.run([self.logits, self.loss_r], feed_dict={
		self.user1: datainput1[0], self.item1: datainput1[1], self.label1: datainput1[2], \
		self.user2: datainput2[0], self.item2: datainput2[1], self.label2: datainput2[2], \
		self.u_his: u_hisinput, self.u_pos: u_posinput, self.u_his_l: u_his_l, self.i_his: i_hisinput, self.i_pos: i_posinput, self.i_his_l: i_his_l, \
		self.isTrain: 0, self.kp: 1,
		})
		return logits, loss_r

	def save(self, sess, path):
		saver = tf.train.Saver()
		saver.save(sess, save_path=path)

	def restore(self, sess, path):
		saver = tf.train.Saver()
		saver.restore(sess, save_path=path)
