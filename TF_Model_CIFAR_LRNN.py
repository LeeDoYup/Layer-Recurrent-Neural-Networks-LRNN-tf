import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import math
import sys
import pdb

from LRNN_utils import *

from config_CIFAR10 import *
from read_CIFAR10 import *
from result_summary import *

class Model(object):
	def __init__(self, sess,x_train,y_train):
		print "Initialize New Model"
		self.sess = sess
		self.train_epoch = 0
		self.train_loss = []
		self.validation_loss = []
		self.min_valid_loss = 9999.999

                train_size = int(len(x_train)*TRAIN_RATE)
                
                self.x_train = x_train[:train_size,:]
                self.y_train = y_train[:train_size,:]
                self.x_valid = x_train[train_size:,:]
                self.y_valid = y_train[train_size:,:]
            

	def _create_place_holders(self):
		#Make Placeholders in order for feed_dict
		self.x_inputs = tf.placeholder(tf.float32,[None, INPUT_DIM],name='x_inputs')
		self.y_inputs = tf.placeholder(tf.float32,[None, OUTPUT_DIM],name='y_inputs')



	def _inference(self):
		#How the Inference Graph (Main Model) is decribed?
		#If you want to change the model, change this part and use proper loss fucntion.
		print "Create Inference Graph of Main Model"

		self.paras={}
		self.opt_paras={}
		self.layers={}
		images = tf.reshape(self.x_inputs, shape=[-1,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_DEPTH])

		with tf.variable_scope('CONV1') as scope:
			self.paras['W1'] = weight_variable('W1',[3,3,3,64])
			self.paras['b1'] = bias_variable('b1',[64])
			self.layers['conv1'] = tf.nn.conv2d(images,self.paras['W1'],strides=[1,1,1,1],padding='SAME')
			self.layers['relu_conv1'] =  tf.nn.relu(self.layers['conv1']+self.paras['b1'], name=scope.name)

                ##########################                  conv1                  ####################################
                with tf.variable_scope('CNN1') as scope:
                        self.layers['CNN1'] = CNN_module(self.layers['relu_conv1'],'Concatnate',64,64)

                with tf.variable_scope('CNN2') as scope:
                        self.layers['CNN2'] = CNN_module(self.layers['CNN1'],'Concatnate',128,64)

                with tf.variable_scope('CNN3') as scope:
                        self.layers['CNN3'] = CNN_module(self.layers['CNN2'],'Concatnate',192,64)

                #######################################################################################################

                with tf.variable_scope('pool1') as scope:
                        self.layers['pool1'] = tf.nn.max_pool(self.layers['CNN3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

                ##########################                  pool 1               ######################################

                with tf.variable_scope('LRNN1') as scope:
                        self.layers['LRNN1'] = LRNN_module(self.layers['pool1'],'Sum','Sum','Forward',256,128)

                with tf.variable_scope('CNN4') as scope:
                        self.layers['CNN4'] = CNN_module(self.layers['LRNN1'],'Concatnate',128,64)

                with tf.variable_scope('LRNN2') as scope:
                        self.layers['LRNN2'] = LRNN_module(self.layers['CNN4'],'Sum','Sum','Forward',192,128)

                with tf.variable_scope('CNN5') as scope:
                        self.layers['CNN5'] = CNN_module(self.layers['LRNN2'],'Concatnate',128,64)

                ########################################################################################################

                with tf.variable_scope('pool2') as scope:
                        self.layers['pool2'] = tf.nn.max_pool(self.layers['CNN5'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

                ##########################                  pool 2               ######################################

                with tf.variable_scope('LRNN3') as scope:
                        self.layers['LRNN3'] = LRNN_module(self.layers['pool2'],'Sum','Sum','Forward',192,128)
 
                with tf.variable_scope('CNN6') as scope:
                        self.layers['CNN6'] = CNN_module(self.layers['LRNN3'],'Concatnate',128,64)
  
                with tf.variable_scope('LRNN4') as scope:
                        self.layers['LRNN4'] = LRNN_module(self.layers['CNN6'],'Sum','Sum','Forward',192,128)

                with tf.variable_scope('CNN7') as scope:
                        self.layers['CNN7'] = CNN_module(self.layers['LRNN4'],'Concatnate',128,64)

               #########################################################################################################
 
                with tf.variable_scope('pool3') as scope:
                        self.layers['pool3'] = tf.nn.max_pool(self.layers['CNN7'],ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')

                with tf.variable_scope('dropout') as scope:
                        self.layers['dropout'] = tf.nn.dropout(self.layers['pool3'],0.5)

                with tf.variable_scope('softmax') as scope:
                        self.paras['W_out'] = weight_variable('W_out',[192,10])
                        self.paras['b_out'] = bias_variable('b_out',[10])
                        self.layers['output'] = full_connected(tf.reshape(self.layers['dropout'],[-1,192]),self.paras['W_out'],self.paras['b_out'],activation ='Softmax')


	def _create_loss(self,loss_type='cross_entropy',regularization='L2'):
		print "Create Loss Function"
		self.loss =  loss_function(self.y_inputs, self.layers['output'], loss_type, regularization, REGULARIZATION_PARA)

	#Types of optimizer = 'GD(Gradient Descent)', 'ADAM', and 'AdaDelta'
	def _create_optimizer(self,type_optimizer='GD',learning_rate = LEARNING_RATE,rate_decaying=IS_RATE_DECAYING, decaying_epoch=DECAYING_EPOCH, decaying_rate=DECAYING_RATE, staircase=STAIRCASE):
		print "Create Optimizer..."
		self.global_step = tf.Variable(0,trainable=False)
		if rate_decaying == False:
			if type_optimizer=='GD':
				self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
			if type_optimizer=='ADAM':
				self.optimizer = tf.train.AdamOptimizer(learning_rate)
			self.train_step = self.optimizer.minimize(self.loss)
			return
		else:
			decaying_learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, int(len(self.x_train)/BATCH_SIZE)*decaying_epoch, decaying_rate, staircase)
			if type_optimizer == 'GD':
				self.optimizer = tf.train.GradientDescentOptimizer(decaying_learning_rate)
			if type_optimizer == 'ADAM':
				self.optimizer = tf.train.AdamOptimizer(decaying_learning_rate)
			self.train_step = self.optimizer.minimize(self.loss)
			return


	#Run Optimizer in order to minimize loss function and return its loss_value
	def batch_train(self,feed_dict):
		loss = self.loss
		loss_value, _ = self.sess.run([loss,self.train_step],feed_dict = feed_dict)
		return loss_value

	#return loss value of feed_dict input
	def batch_loss(self,feed_dict):
		loss = self.loss
		loss_value = self.sess.run(loss,feed_dict=feed_dict)
		return loss_value

	#return output result of feed_dict input
	def batch_out(self,feed_dict):
		outputs = self.sess.run(self.layers['output'], feed_dict=feed_dict)
		return outputs


	def run_train_epoch(self):
		#parameter initializes: loss, cumulative loss
		loss = 0
		cumulative_loss = 0
		train_loss = 0
		self.train_epoch += 1

		batch_number = int(len(self.x_train)/BATCH_SIZE)

		for i in range(batch_number):
			x_batch = self.x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
			y_batch = self.y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]

			feed_dict = {self.x_inputs:x_batch, self.y_inputs:y_batch}

			loss = self.batch_train(feed_dict)
			cumulative_loss += loss
			#calculate average_loss for display
			average_loss = cumulative_loss / (i+1)

			sys.stdout.write("\r training loss : "+str(average_loss)+" | "+str(i+1)+"th/"+str(batch_number)+"batches")
			sys.stdout.flush()
			train_loss += loss

		train_loss = train_loss/batch_number
		self.train_loss.append(train_loss)
		return train_loss

	def run_validation(self):
		valid_size = len(self.x_valid)
                cumulative_loss = 0
                loss = 0
                valid_loss = 0
                batch_number = int(valid_size/BATCH_SIZE)
                for i in range(batch_number):
                  x_batch = self.x_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                  y_batch = self.y_valid[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]
                  
                  feed_dict = {self.x_inputs:x_batch, self.y_inputs:y_batch}
 
                  loss = self.batch_loss(feed_dict)
                  cumulative_loss += loss
                  average_loss = cumulative_loss/(i+1)
         
                  sys.stdout.write("\r validation loss : "+str(average_loss)+" | "+str(i+1)+"th/"+str(batch_number)+"batches")
                  sys.stdout.flush()
                  valid_loss += loss
                
		valid_loss = valid_loss/batch_number
		self.validation_loss.append(valid_loss)
		print "\n Validation loss:  ", valid_loss
                
		if self.min_valid_loss > valid_loss:
			self.update_optimal_variables()

		return valid_loss


	def use_optimal_paras(self):
		for i in self.paras.keys():
			self.paras[i] = self.opt_paras[i]

	def run_test(self, use_optimal_parameter=True):
		if use_optimal_parameter == True:
			self.use_optimal_paras()
                
                self.x_test , self.y_test = get_CIFAR_data(False)
		test_size = len(self.x_test)
		feed_dict = {self.x_inputs: self.x_test, self.y_inputs: self.y_test}
		self.test_loss = self.batch_loss(feed_dict)
		self.test_out = self.batch_out(feed_dict)
		print "Test loss:  ", self.test_loss
		return self.test_loss, self.test_out

	def build_graph(self):
		self._create_place_holders()
		self._inference()
		self._create_loss()
		self._create_optimizer(OPTIMIZER_TYPE)

	def initialize_variables(self):
		init = tf.global_variables_initializer()
		sess.run(init)


	def update_optimal_variables(self):
		for i in self.paras.keys():
			self.opt_paras[i] = self.paras[i]


	def run(self):
		self.build_graph()
		self.initialize_variables()
		for i in range(EPOCH):
			self.run_train_epoch()
			self.run_validation()
			print "\n"
		self.run_test(True)
		test_accuracy = classification_accuracy(self.test_out, self.y_test)
		np.save('test_out.npy',self.test_out)
		np.save('test_loss.npy',self.test_loss)
		np.save('test_accuracy.npy',test_accuracy)

		saver = tf.train.Saver()
		save_path = saver.save(self.sess, "LRNN.ckpt")
		print("Model saved in file: %s" % save_path)

		for i in self.paras.keys():
			np.save(i+'.npy', self.sess.run(self.paras[i]))
			print "\n Save Parameter: ", i
		
		feed_dict = {self.x_inputs: self.x_train, self.y_inputs: self.y_train}
		self.train_out = self.batch_out(feed_dict)
		np.save('train_out.npy',self.train_out)

		np.save('train_loss.npy',self.train_loss)
		np.save('validation_loss.npy',self.validation_loss)
		pdb.set_trace()
		return

sess = tf.Session()
x_train , y_train = get_CIFAR_data(True) 
model1  = Model(sess,x_train,y_train)
model1.run()












