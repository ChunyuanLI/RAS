from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
import tensorflow as tf
from ops import linear
from ops import gmm_sample
from utils import make_batches
from utils import disp_scatter

from tqdm import tqdm

class GAN(object):
	"""Dual Discriminator Generative Adversarial Nets for 2D data
	"""

	def __init__(self,
				 sample_n,
				 model_name="d2gan",
				 anneal=0,
				 num_z=2,  # number of noise variables
				 hidden_size=128,
				 alpha=1.0,  # coefficient - regularization constant of D1
				 beta=1.0,  # coefficient - regularization constant of D2
				 mix_coeffs=(0.5, 0.5),
				 mean=((0.0, 0.0), (1.0, 1.0)),
				 cov=((0.1, 0.1), (0.1, 0.1)),
				 batch_size=512,
				 learning_rate=0.0002,
				 num_epochs=25000,
				 disp_freq=5000,
				 disp_flag=True,
				 random_seed=6789,
				 ):
		self.model_name = model_name
		self.anneal=anneal
		self.num_z = num_z
		self.hidden_size = hidden_size
		self.alpha = alpha
		self.beta = beta
		self.mix_coeffs = mix_coeffs
		self.mean = mean
		self.cov = cov
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs+1
		self.disp_freq = disp_freq
		self.disp_flag=disp_flag
		self.random_seed = random_seed
		self.points=[]
		self.n_sample=sample_n
		self.sample=gmm_sample(self.n_sample, self.mix_coeffs, self.mean, self.cov)
		self.n_sample=self.sample.shape[0]
		print(self.anneal)
	def _init(self):
		self.epoch = 0
		self.fig = None
		self.ax = None

		# TensorFlow's initialization
		self.tf_graph = tf.Graph()
		self.tf_config = tf.ConfigProto()
		self.tf_config.gpu_options.allow_growth = True
		self.tf_config.log_device_placement = False
		self.tf_config.allow_soft_placement = True
		self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)

		np.random.seed(self.random_seed)
		with self.tf_graph.as_default():
			tf.set_random_seed(self.random_seed)

	def _build_model(self):
		# This defines the generator network - it takes samples from a noise
		# distribution as input, and passes them through an MLP.
		# print(np.random.randn(10))
		self.anneal_factor=tf.constant(1.0)
		if self.model_name=='ALLgan_cc':
			with tf.variable_scope('generator'):
				self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
				self.x_g = self._create_generator(self.z, self.hidden_size)
				self.x = tf.placeholder(tf.float32, shape=[None, 2])
				self.z_rec=self._create_inference(self.x_g,self.hidden_size)
			with tf.variable_scope('discriminator_1') as scope:
				self.d1x = self._create_discriminator(self.x, self.hidden_size)
				scope.reuse_variables()
				self.d1g = self._create_discriminator(self.x_g, self.hidden_size)
			
			with tf.variable_scope('discriminator_2') as scope:
				self.d2x = self._create_discriminator(self.x, self.hidden_size)
				scope.reuse_variables()
				self.d2g = self._create_discriminator(self.x_g, self.hidden_size)

			self.d1_loss = tf.reduce_mean(self.alpha * tf.nn.softplus(self.d1x) + tf.nn.softplus(-self.d1g) )
			self.d2_loss = tf.reduce_mean(tf.nn.softplus(self.d2x) + self.beta * tf.nn.softplus(-self.d2g) )
			self.d_loss = self.d1_loss+0.0*self.d2_loss
			self.g2r=tf.reduce_mean(self.d1g)
			self.z_cost = tf.reduce_mean(tf.pow(self.z_rec - self.z, 2))
			self.g_loss = self.g2r + self.anneal_factor*self.z_cost 
		elif self.model_name=='gan_cc':
			with tf.variable_scope('generator'):
				self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
				self.x_g = self._create_generator(self.z, self.hidden_size)
				self.x = tf.placeholder(tf.float32, shape=[None, 2])
				self.z_rec=self._create_inference(self.x_g,self.hidden_size)
			with tf.variable_scope('discriminator_1') as scope:
				self.d1x = self._create_discriminator(self.x, self.hidden_size)
				scope.reuse_variables()
				self.d1g = self._create_discriminator(self.x_g, self.hidden_size)
			
			with tf.variable_scope('discriminator_2') as scope:
				self.d2x = self._create_discriminator(self.x, self.hidden_size)
				scope.reuse_variables()
				self.d2g = self._create_discriminator(self.x_g, self.hidden_size)

			self.d1_loss = tf.reduce_mean(self.alpha * tf.nn.softplus(-self.d1x) + tf.nn.softplus(self.d1g) )
			self.d2_loss = tf.reduce_mean(tf.nn.softplus(self.d2x) + self.beta * tf.nn.softplus(-self.d2g) )
			self.d_loss = self.d1_loss+0.0*self.d2_loss
			self.g2r=tf.reduce_mean(tf.nn.softplus(-self.d1g))
			self.z_cost = tf.reduce_mean(tf.pow(self.z_rec - self.z, 2))
			self.g_loss = self.g2r + self.anneal_factor*self.z_cost 
		elif self.model_name=='SNgan_cc':
			with tf.variable_scope('generator'):
				self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
				self.x_g = self._create_generator(self.z, self.hidden_size)
				self.x = tf.placeholder(tf.float32, shape=[None, 2])
				self.z_rec=self._create_inference(self.x_g,self.hidden_size)
			with tf.variable_scope('discriminator_1') as scope:
				self.d1x = self._create_discriminator(self.x, self.hidden_size,use_SN=True,update_collection=None)
				scope.reuse_variables()
				self.d1g = self._create_discriminator(self.x_g, self.hidden_size,use_SN=True,update_collection='NO_OPS')
			
			with tf.variable_scope('discriminator_2') as scope:
				self.d2x = self._create_discriminator(self.x, self.hidden_size)
				scope.reuse_variables()
				self.d2g = self._create_discriminator(self.x_g, self.hidden_size)

			self.d1_loss = tf.reduce_mean(self.alpha * tf.nn.softplus(-self.d1x) + tf.nn.softplus(self.d1g) )
			self.d2_loss = tf.reduce_mean(tf.nn.softplus(self.d2x) + self.beta * tf.nn.softplus(-self.d2g) )
			self.d_loss = self.d1_loss+0.0*self.d2_loss
			self.g2r=tf.reduce_mean(tf.nn.softplus(-self.d1g))
			self.z_cost = tf.reduce_mean(tf.pow(self.z_rec - self.z, 2))
			self.g_loss = self.g2r + self.z_cost 
		elif self.model_name=='d2gan_cc':
			with tf.variable_scope('generator'):
				self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
				self.x_g = self._create_generator(self.z, self.hidden_size)
				self.x = tf.placeholder(tf.float32, shape=[None, 2])
				self.z_rec=self._create_inference(self.x_g,self.hidden_size)

			with tf.variable_scope('discriminator_1') as scope:
				self.d1x = self._create_discriminator_d2(self.x, self.hidden_size)
				scope.reuse_variables()
				self.d1g = self._create_discriminator_d2(self.x_g, self.hidden_size)
			with tf.variable_scope('discriminator_2') as scope:
				self.d2x = self._create_discriminator_d2(self.x, self.hidden_size)
				scope.reuse_variables()
				self.d2g = self._create_discriminator_d2(self.x_g, self.hidden_size)

	   
			self.d1_loss = tf.reduce_mean(-self.alpha * tf.log(self.d1x) + self.d1g)
			self.d2_loss = tf.reduce_mean(self.d2x - self.beta * tf.log(self.d2g))
			self.d_loss = self.d1_loss + self.d2_loss
			self.z_cost = tf.reduce_mean(tf.pow(self.z_rec - self.z, 2))
			self.g_loss = tf.reduce_mean(-self.d1g + self.beta * tf.log(self.d2g))+self.z_cost
		else:
			with tf.variable_scope('generator'):
				self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
				self.x_g = self._create_generator(self.z, self.hidden_size)

			self.x = tf.placeholder(tf.float32, shape=[None, 2])
			# The discriminator tries to tell the difference between samples from the
			# true data distribution (self.x) and the generated samples (self.z).
			#
			# Here we create two copies of the discriminator network (that share parameters),
			# as you cannot use the same network with different inputs in TensorFlow.
			if self.model_name=='d2gan':
				with tf.variable_scope('discriminator_1') as scope:
					self.d1x = self._create_discriminator_d2(self.x, self.hidden_size)
					scope.reuse_variables()
					self.d1g = self._create_discriminator_d2(self.x_g, self.hidden_size)
				with tf.variable_scope('discriminator_2') as scope:
					self.d2x = self._create_discriminator_d2(self.x, self.hidden_size)
					scope.reuse_variables()
					self.d2g = self._create_discriminator_d2(self.x_g, self.hidden_size)

		   
				self.d1_loss = tf.reduce_mean(-self.alpha * tf.log(self.d1x) + self.d1g)
				self.d2_loss = tf.reduce_mean(self.d2x - self.beta * tf.log(self.d2g))
				self.d_loss = self.d1_loss + self.d2_loss
				self.g_loss = tf.reduce_mean(-self.d1g + self.beta * tf.log(self.d2g))
			elif self.model_name=='SNgan':
				print('SNgan')
				with tf.variable_scope('discriminator_1') as scope:
						self.d1x = self._create_discriminator(self.x, self.hidden_size,use_SN=True,update_collection=None)
						scope.reuse_variables()
						self.d1g = self._create_discriminator(self.x_g, self.hidden_size,use_SN=True,update_collection='NO_OPS')
				with tf.variable_scope('discriminator_2') as scope:
					self.d2x = self._create_discriminator(self.x, self.hidden_size)
					scope.reuse_variables()
					self.d2g = self._create_discriminator(self.x_g, self.hidden_size)
				self.d1_loss = tf.reduce_mean(self.alpha * tf.nn.softplus(-self.d1x) + tf.nn.softplus(self.d1g) )
				self.d2_loss = tf.reduce_mean(tf.nn.softplus(self.d2x) + self.beta * tf.nn.softplus(-self.d2g) )
				self.d_loss = self.d1_loss + 0.0*self.d2_loss
				self.g_loss = tf.reduce_mean(tf.nn.softplus(-self.d1g) + 0.0 * self.d2g)
			else:
				with tf.variable_scope('discriminator_1') as scope:
					self.d1x = self._create_discriminator(self.x, self.hidden_size)
					scope.reuse_variables()
					self.d1g = self._create_discriminator(self.x_g, self.hidden_size)
				with tf.variable_scope('discriminator_2') as scope:
					self.d2x = self._create_discriminator(self.x, self.hidden_size)
					scope.reuse_variables()
					self.d2g = self._create_discriminator(self.x_g, self.hidden_size)

				if self.model_name=='gan':
					print('gan')
					self.d1_loss = tf.reduce_mean(self.alpha * tf.nn.softplus(-self.d1x) + tf.nn.softplus(self.d1g) )
					self.d2_loss = tf.reduce_mean(tf.nn.softplus(self.d2x) + self.beta * tf.nn.softplus(-self.d2g) )
					self.d_loss = self.d1_loss + 0.0*self.d2_loss
					self.g_loss = tf.reduce_mean(tf.nn.softplus(-self.d1g) + 0.0 * self.d2g)
				elif self.model_name=='ALLgan':
					print('ALLgan')
					self.d1_loss = tf.reduce_mean(self.alpha * tf.nn.softplus(self.d1x) + tf.nn.softplus(-self.d1g) )
					self.d2_loss = tf.reduce_mean(tf.nn.softplus(self.d2x) + self.beta * tf.nn.softplus(-self.d2g) )
					self.d_loss = self.d1_loss + 0.0*self.d2_loss
					self.g_loss = tf.reduce_mean(self.d1g + 0.0 * self.d2g)
				else:
					print('avaiable models are: d2gan,gan,ALLgan,SNgan,d2gan_cc,gan_cc,ALLgan_cc,SNgan_cc')

		self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
										  scope='discriminator_1') \
						+ tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
											scope='discriminator_2')
		self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

		self.d_opt = self._create_optimizer(self.d_loss, self.d_params,
											self.learning_rate)
		self.g_opt = self._create_optimizer(self.g_loss, self.g_params,
											self.learning_rate)
	def _create_inference(self, input, h_dim):
		hidden = tf.nn.relu(linear(input, h_dim, 'g_hidden3'))
		hidden = tf.nn.relu(linear(hidden, h_dim, 'g_hidden4'))
		out =linear(hidden, 2, scope='g_out_')
		return out
	def _create_inference_noise(self, input1,input2, h_dim):
		input = tf.concat([input1, input2], axis=1)
		hidden = tf.nn.relu(linear(input, h_dim, 'g_hidden3'))
		hidden = tf.nn.relu(linear(hidden, h_dim, 'g_hidden4'))
		out =linear(hidden, 2, scope='g_out_')
		return out

	def _create_generator(self, input, h_dim):
		hidden = tf.nn.relu(linear(input, h_dim, 'g_hidden1'))
		hidden = tf.nn.relu(linear(hidden, h_dim, 'g_hidden2'))
		out = linear(hidden, 2, scope='g_out')
		return out

	def _create_discriminator(self, input, h_dim,use_SN=False,update_collection=None):
		hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden1',use_SN=use_SN,update_collection=update_collection ))
		out = linear(hidden, 1, scope='d_out',use_SN=use_SN,update_collection=update_collection)
		return out

	def _create_pair_discriminator(self, input1, input2, h_dim):
		input = tf.concat([input1, input2], axis=1)
		hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden1', ))
		out = linear(hidden, 1, scope='d_out')
		return out

	def _create_discriminator_d2(self, input, h_dim):
		#special discriminator for d2gan
		hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden1', ))
		out = tf.nn.softplus(linear(hidden, 1, scope='d_out'))
		return out



	def _create_optimizer(self, loss, var_list, initial_learning_rate):
		return tf.train.AdamOptimizer(initial_learning_rate,
									  beta1=0.5).minimize(loss, var_list=var_list)

	def fit(self):
		if (not hasattr(self, 'epoch')) or self.epoch == 0:
			self._init()
			with self.tf_graph.as_default():
				self._build_model()
				self.tf_session.run(tf.global_variables_initializer())
		anneal_factor=1.0
		for i in tqdm(range(self.num_epochs)):
			# update discriminator
			if self.anneal==True:
				if i>10000:
					anneal_factor=max(0.0,anneal_factor-0.00005)
			if self.batch_size>self.n_sample:
				x=np.zeros((self.batch_size,2))
				mode_n=int(self.n_sample/8)
				batch_n=int(self.batch_size/8)
				for i in range(8):
					x[batch_n*i:batch_n*(i+1),:]=self.sample[np.random.choice(mode_n,batch_n,replace=True)+mode_n*i,:]
			else:
				x=np.zeros((self.batch_size,2))
				mode_n=int(self.n_sample/8)
				batch_n=int(self.batch_size/8)
				for i in range(8):
					x[batch_n*i:batch_n*(i+1),:]=self.sample[np.random.choice(mode_n,batch_n,replace=False)+mode_n*i,:]
			z = np.random.normal(0.0, 1.0, [self.batch_size, self.num_z])
			if self.model_name=='svaer_noise':
				n1=np.random.normal(0.0, 1.0, [self.batch_size, self.num_z])
				n2=np.random.normal(0.0, 1.0, [self.batch_size, self.num_z])
				_ = self.tf_session.run(
				[self.d_opt],
				feed_dict={self.x: np.reshape(x, [self.batch_size, 2]),
						   self.z: np.reshape(z, [self.batch_size, self.num_z]),
						   self.noise1: np.reshape(n1, [self.batch_size, self.num_z]),
						   self.noise2: np.reshape(n2, [self.batch_size, self.num_z]),
						   })
			else:
				_ = self.tf_session.run(
					[self.d_opt],
					feed_dict={self.x: np.reshape(x, [self.batch_size, 2]),
							   self.z: np.reshape(z, [self.batch_size, self.num_z]),
							   self.anneal_factor:anneal_factor
							   })
			# d1x, d2x, d1_loss, d2_loss, d_loss, _ = self.tf_session.run(
			# 	[self.d1x, self.d2x, self.d1_loss, self.d2_loss, self.d_loss, self.d_opt],
			# 	feed_dict={self.x: np.reshape(x, [self.batch_size, 2]),
			# 			   self.z: np.reshape(z, [self.batch_size, self.num_z]),
			# 			   })
			# update generator
			if self.batch_size>self.n_sample:
				x=np.zeros((self.batch_size,2))
				mode_n=int(self.n_sample/8)
				batch_n=int(self.batch_size/8)
				for i in range(8):
					x[batch_n*i:batch_n*(i+1),:]==self.sample[np.random.choice(mode_n,batch_n,replace=True)+mode_n*i,:]
			else:
				x=np.zeros((self.batch_size,2))
				mode_n=int(self.n_sample/8)
				batch_n=int(self.batch_size/8)
				for i in range(8):
					x[batch_n*i:batch_n*(i+1),:]==self.sample[np.random.choice(mode_n,batch_n,replace=False)+mode_n*i,:]
			z = np.random.normal(0.0, 1.0, [self.batch_size, self.num_z])
			if self.model_name=='svaer_noise':
				n1=np.random.normal(0.0, 1.0, [self.batch_size, self.num_z])
				n2=np.random.normal(0.0, 1.0, [self.batch_size, self.num_z])
				g_loss, _ = self.tf_session.run(
				[self.g_loss, self.g_opt],
				feed_dict={self.x: np.reshape(x, [self.batch_size, 2]),
						   self.z: np.reshape(z, [self.batch_size, self.num_z]),
						   self.noise1: np.reshape(n1, [self.batch_size, self.num_z]),
						   self.noise2: np.reshape(n2, [self.batch_size, self.num_z])
						   })
			else:
				g_loss, _ = self.tf_session.run(
					[self.g_loss, self.g_opt],
					feed_dict={self.x: np.reshape(x, [self.batch_size, 2]),
							   self.z: np.reshape(z, [self.batch_size, self.num_z]),
							   self.anneal_factor:anneal_factor
							   })
			# if self.epoch%5000==0:
			#     print("Epoch: [%4d/%4d] d1_loss: %.8f, d2_loss: %.8f,"
			#           " d_loss: %.8f, g_loss: %.8f" % (self.epoch, self.num_epochs,
			#                                            d1_loss, d2_loss, d_loss, g_loss))
			if self.epoch % self.disp_freq == 0:
				self.display(num_samples=10000)
			self.epoch += 1

	def generate(self, num_samples=1000):
		zs = np.random.normal(0.0, 1.0, [num_samples, self.num_z])
		g = np.zeros([num_samples, 2])
		batches = make_batches(num_samples, self.batch_size)
		for batch_idx, (batch_start, batch_end) in enumerate(batches):
			g[batch_start:batch_end] = self.tf_session.run(
				self.x_g,
				feed_dict={
					self.z: np.reshape(zs[batch_start:batch_end],
									   [batch_end - batch_start, self.num_z])
				}
			)
		return g

	def display(self, num_samples=20000):
		x = self.sample
		g = self.generate(num_samples=num_samples)
		if self.disp_flag:
			print('ploting!')
			self.fig = None
			self.ax = None        
			self.fig, self.ax = disp_scatter(x, g, fig=self.fig, ax=self.ax)
			self.fig.tight_layout()
			if self.anneal:
				dir_disp = 'output_'+self.model_name+'anneal'+'%d'%self.n_sample
			else:
				dir_disp = 'output_'+self.model_name+'%d'%self.n_sample
			if not os.path.exists(dir_disp):
				os.makedirs(dir_disp)
			print(dir_disp)
			self.fig.savefig(dir_disp + '/'+'%d_'%(self.random_seed)+"{}.png".format(self.epoch))
		
		self.points.append(g)

	def save_point(self):
		if self.anneal:
			dir_disp = 'output_'+self.model_name+'anneal'+'%d'%self.n_sample
		else:
			dir_disp = 'output_'+self.model_name+'%d'%self.n_sample
		if not os.path.exists(dir_disp):
			os.makedirs(dir_disp)
		np.save(dir_disp + "/%d"%(self.random_seed),self.points)
		np.save(dir_disp + "/%d"%(self.random_seed)+'target_samples',self.sample)
