import tensorflow as tf

class Model(object):
    
	def __init__(self, user_count, item_count):
 
		self.user = tf.placeholder(tf.int32, [None,]) # [B]
		self.item = tf.placeholder(tf.int32, [None,]) # [B]
		self.label = tf.placeholder(tf.float32, [None,]) # [B]
		self.u_his = tf.placeholder(tf.int32, [None, None]) # [B, I]
		self.u_his_l = tf.placeholder(tf.float32, [None,]) # [B]
		self.i_his = tf.placeholder(tf.int32, [None, None]) # [B, U]
		self.i_his_l = tf.placeholder(tf.float32, [None,]) # [B]

		self.lr = tf.placeholder(tf.float32)
		self.reg = tf.placeholder(tf.float32) 
		self.isTrain = tf.placeholder(tf.int32)
		self.kp = tf.placeholder(tf.float32)

		#--------------embedding layer-------------------
         
		hidden_units_u = 20
		hidden_units_i = 20

		user_emb = tf.get_variable("u_user_emb", [user_count, hidden_units_u])
		item_emb = tf.get_variable("i_item_emb", [item_count, hidden_units_i])
		user_b = tf.get_variable("u_user_b", [user_count],
                             initializer=tf.constant_initializer(0.0))
		item_b = tf.get_variable("i_item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))

		u_emb = tf.nn.embedding_lookup(user_emb, self.user)
		i_emb = tf.nn.embedding_lookup(item_emb, self.item)
		u_b = tf.gather(user_b, self.user)
		i_b = tf.gather(item_b, self.item)
		u_his_emb = tf.nn.embedding_lookup(item_emb, self.u_his)
		i_his_emb = tf.nn.embedding_lookup(user_emb, self.i_his)

		#--------------network-------------------

		def sequence_mask(object_, key):
			key_masks = tf.sequence_mask(key, tf.shape(object_)[1])
			key_masks = tf.expand_dims(key_masks, axis = 2)
			key_masks = tf.tile(key_masks, [1, 1, tf.shape(object_)[2]])
			key_masks = tf.reshape(key_masks, [-1, tf.shape(object_)[1], tf.shape(object_)[2]])
			paddings = tf.zeros_like(object_)
			output = tf.where(key_masks, object_, paddings)
			return output
		u_his_emb = sequence_mask(u_his_emb, self.u_his_l)
		i_his_emb = sequence_mask(i_his_emb, self.i_his_l)
		
		def RNN(x, hidden_num, name):
			x = tf.reshape(x, shape=[-1, tf.shape(x)[1], hidden_num])
			rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num, name=name)
			output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
			return states
		i_d_emb = RNN(u_his_emb, hidden_units_i, 'i_rnn_l1')
		i_d_emb = tf.reshape(i_d_emb, [-1, hidden_units_i])
		u_d_emb = RNN(i_his_emb, hidden_units_u, 'u_rnn_l1')
		u_d_emb = tf.reshape(u_d_emb, [-1, hidden_units_u])

		i_nn_1 = tf.layers.dense(i_d_emb, 1, activation=None, use_bias = True, name='i_nn_l1')
		u_nn_1 = tf.layers.dense(u_d_emb, 1, activation=None, use_bias = True, name='u_nn_l1')
		self.logits = tf.reduce_sum(tf.multiply(i_nn_1, u_nn_1), axis=1) + u_b + i_b \
		+ tf.reduce_sum(tf.multiply(u_emb, i_emb), axis=1)
		self.logits = tf.reshape(self.logits, [-1])
		
		#--------------loss computation-------------------
		
		self.loss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits))
		self.loss_reg = tf.reduce_sum(tf.square(u_emb)) + tf.reduce_sum(tf.square(i_emb)) \
		+ tf.reduce_sum(tf.square(u_his_emb)) + tf.reduce_sum(tf.square(i_his_emb))
		self.loss = self.loss_r + self.reg*self.loss_reg

		# Step variable
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.global_epoch_step = \
		tf.Variable(0, trainable=False, name='global_epoch_step')
		self.global_epoch_step_op = \
		tf.assign(self.global_epoch_step, self.global_epoch_step+1)

		self.opt1 = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		#self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
		trainable_params1 = tf.trainable_variables('u')
		gradients1 = tf.gradients(self.loss, trainable_params1)
		clip_gradients1, _ = tf.clip_by_global_norm(gradients1, 20*self.lr)
		self.train_op1 = self.opt1.apply_gradients(zip(clip_gradients1, trainable_params1), global_step=self.global_step)

		self.opt2 = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		#self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
		trainable_params2 = tf.trainable_variables('i')
		gradients2 = tf.gradients(self.loss, trainable_params2)
		clip_gradients2, _ = tf.clip_by_global_norm(gradients2, 20*self.lr)
		self.train_op2 = self.opt2.apply_gradients(zip(clip_gradients2, trainable_params2))
		
		#--------------end model---------------

	def train(self, sess, datainput, u_hisinput, u_his_l, i_hisinput, i_his_l, lr, kp, reg):
		loss_r, loss_reg, _ = sess.run([self.loss_r, self.loss_reg, self.train_op1], feed_dict={
		self.user: datainput[0], self.item: datainput[1], self.label: datainput[2], \
		self.u_his: u_hisinput, self.u_his_l: u_his_l, self.i_his: i_hisinput, self.i_his_l: i_his_l, \
		self.lr: lr, self.reg: reg, self.isTrain: 1, self.kp: kp,
		})
		loss_r, loss_reg, _ = sess.run([self.loss_r, self.loss_reg, self.train_op2], feed_dict={
		self.user: datainput[0], self.item: datainput[1], self.label: datainput[2], \
		self.u_his: u_hisinput, self.u_his_l: u_his_l, self.i_his: i_hisinput, self.i_his_l: i_his_l, \
		self.lr: lr, self.reg: reg, self.isTrain: 1, self.kp: kp,
		})
		return loss_r, loss_reg

	def eval(self, sess, datainput, u_hisinput, u_his_l, i_hisinput, i_his_l):
		logits, loss_r = sess.run([self.logits, self.loss_r], feed_dict={
		self.user: datainput[0], self.item: datainput[1], self.label: datainput[2], \
		self.u_his: u_hisinput, self.u_his_l: u_his_l, self.i_his: i_hisinput, self.i_his_l: i_his_l, \
		self.isTrain: 0, self.kp: 1,
		})
		return logits, loss_r

	def save(self, sess, path):
		saver = tf.train.Saver()
		saver.save(sess, save_path=path)

	def restore(self, sess, path):
		saver = tf.train.Saver()
		saver.restore(sess, save_path=path)
