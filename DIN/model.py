import tensorflow as tf

class Model(object):
    
	def __init__(self, user_count, item_count):
 
		self.user = tf.placeholder(tf.int32, [None,]) # [B]
		self.item = tf.placeholder(tf.int32, [None,]) # [B]
		self.label = tf.placeholder(tf.float32, [None,]) # [B]
		self.u_his = tf.placeholder(tf.int32, [None, None]) # [B, I]
		self.u_pos = tf.placeholder(tf.int32, [None, None]) # [B, I]
		self.u_his_l = tf.placeholder(tf.float32, [None,]) # [B]

		self.lr = tf.placeholder(tf.float32)
		self.reg = tf.placeholder(tf.float32) 
		self.isTrain = tf.placeholder(tf.int32)
		self.kp = tf.placeholder(tf.float32)

		#--------------embedding layer-------------------
         
		hidden_units_u = 20
		hidden_units_i = 20

		user_emb = tf.get_variable("r_user_emb", [user_count, hidden_units_u])
		item_emb = tf.get_variable("r_item_emb", [item_count, hidden_units_i])
		user_b = tf.get_variable("r_user_b", [user_count],
                             initializer=tf.constant_initializer(0.0))
		item_b = tf.get_variable("r_item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
		pos_emb = tf.get_variable("r_pos_emb", [20, hidden_units_i])

		u_emb = tf.nn.embedding_lookup(user_emb, self.user)
		i_emb = tf.nn.embedding_lookup(item_emb, self.item)
		u_b = tf.gather(user_b, self.user)
		i_b = tf.gather(item_b, self.item)
		u_his_emb = tf.nn.embedding_lookup(item_emb, self.u_his)
		u_pos_emb = tf.nn.embedding_lookup(pos_emb, self.u_pos)

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
		u_pos_emb = sequence_mask(u_pos_emb, self.u_his_l)
		u_his_emb_ = u_his_emb + u_pos_emb
		
		i_emb_ = tf.expand_dims(i_emb, axis=1)
		i_emb_ = tf.tile(i_emb_, [1, tf.shape(u_his_emb_)[1], 1])
		attn_in = tf.concat([i_emb_, u_his_emb_], axis=2)
		weights = tf.layers.dense(attn_in, 1, activation=tf.nn.tanh, use_bias = True, name='attn_l1')
		weights = tf.nn.softmax(weights, axis=1)
		weights_ = tf.tile(weights, [1, 1, hidden_units_i])
		i_d_emb = tf.reduce_sum(tf.multiply(weights_, u_his_emb_), axis=1)

		nn_in = tf.concat([u_emb, i_emb, i_d_emb], axis=1)
		nn_1 = tf.layers.dense(nn_in, 1, activation=None, use_bias = True, name='nn_l1')
		self.logits = tf.reshape(nn_1, [-1])
		
		#--------------loss computation-------------------
		
		self.loss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits))
		self.loss_reg = tf.reduce_sum(tf.square(u_emb)) + tf.reduce_sum(tf.square(i_emb)) \
		+ tf.reduce_sum(tf.square(u_his_emb))
		self.loss = self.loss_r + self.reg*self.loss_reg

		# Step variable
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.global_epoch_step = \
		tf.Variable(0, trainable=False, name='global_epoch_step')
		self.global_epoch_step_op = \
		tf.assign(self.global_epoch_step, self.global_epoch_step+1)

		self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		#self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999)
		trainable_params = tf.trainable_variables()
		gradients = tf.gradients(self.loss, trainable_params)
		clip_gradients, _ = tf.clip_by_global_norm(gradients, 20*self.lr)
		self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)
		
		#--------------end model---------------

	def train(self, sess, datainput, u_hisinput, u_posinput, u_his_l, lr, kp, reg):
		loss_r, loss_reg, _ = sess.run([self.loss_r, self.loss_reg, self.train_op], feed_dict={
		self.user: datainput[0], self.item: datainput[1], self.label: datainput[2], \
		self.u_his: u_hisinput, self.u_pos: u_posinput, self.u_his_l: u_his_l, self.lr: lr, self.reg: reg, \
		self.isTrain: 1, self.kp: kp,
		})
		return loss_r, loss_reg

	def eval(self, sess, datainput, u_hisinput, u_posinput, u_his_l):
		logits, loss_r = sess.run([self.logits, self.loss_r], feed_dict={
		self.user: datainput[0], self.item: datainput[1], self.label: datainput[2], \
		self.u_his: u_hisinput, self.u_pos: u_posinput, self.u_his_l: u_his_l, \
		self.isTrain: 0, self.kp: 1,
		})
		return logits, loss_r

	def save(self, sess, path):
		saver = tf.train.Saver()
		saver.save(sess, save_path=path)

	def restore(self, sess, path):
		saver = tf.train.Saver()
		saver.restore(sess, save_path=path)
