import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import math
import sys
import pdb


def Recurrent_Cell(output_size,cell_type='Basic',activation=tf.nn.relu):
    if cell_type == 'Basic':
      return tf.contrib.rnn.BasicRNNCell(output_size,activation=activation)
    if cell_type == 'LSTM':
      return tf.contrib.rnn.BasicLSTMCell(output_size,activation=activation)
    if cell_type == 'GRU':
      return tf.contrib.rnn.GRUCell(output_size,activation=activation)
    if cell_type == 'LayerNorm':
      return tf.contrib.rnn.LayerNormBasicLSTMCell(output_size,activation=activation)



def Layer_recurrent(inputs,output_size,cell_type='Basic',direction_to='LR', result_op='Sum',activation=tf.nn.relu):
# input shape : [m,n,d] / weight size : [d,D] / weight_recurrent size : [D,D], bias shape [D]
  shape_list = inputs.get_shape().as_list()
  batch_size,width, height = BATCH_SIZE,shape_list[1], shape_list[2]
  if direction_to == 'LR':
     time_step_size = width
     inputs = tf.reshape(inputs,[-1, width , height*shape_list[3]])
     input_reverse = tf.reverse(inputs,axis=[1])
     with tf.variable_scope('L2R') as scope:
         cells = Recurrent_Cell(height*output_size,cell_type,activation)

         outputs1, states  = tf.nn.dynamic_rnn(cells,inputs,dtype=tf.float32)
         outputs1 = tf.reshape(outputs1,[-1,width,height,output_size])

     with tf.variable_scope('R2L') as scope:
         cells = Recurrent_Cell(height*output_size,cell_type,activation)
         
         outputs2, states = tf.nn.dynamic_rnn(cells,input_reverse,dtype=tf.float32)
         outputs2 = tf.reshape(outputs2,[-1,width,height,output_size])

  if direction_to == 'UD':
     time_step_size = height
     input_transpose = tf.transpose(inputs,perm=[0,2,1,3])
     inputs = tf.reshape(input_transpose,[-1,height,width*shape_list[3]])       
     input_reverse = tf.reverse(inputs,axis=[1])
     with tf.variable_scope('U2D') as scope:
         cells = Recurrent_Cell(width*output_size,cell_type,activation)

         outputs1, states  = tf.nn.dynamic_rnn(cells,inputs,dtype=tf.float32)
         outputs1 = tf.reshape(outputs1,[-1,width,height,output_size])
         outputs1 = tf.transpose(outputs1,perm=[0,2,1,3])

     with tf.variable_scope('D2U') as scope:
         cells = Recurrent_Cell(width*output_size,cell_type,activation)
         
         outputs2, states = tf.nn.dynamic_rnn(cells,input_reverse,dtype=tf.float32)
         outputs2 = tf.reshape(outputs2,[-1,width,height,output_size])
         outputs2 = tf.transpose(outputs2,perm=[0,2,1,3])

  if result_op == 'Sum':
    return outputs1+outputs2
  if result_op == 'Concatenate':
    return tf.concat([outputs1,outputs2],3)
        






def weight_variable(name, shape, initializer = 'normal'):
#Random normal initializer
    if initializer == 'normal':
       initial = tf.truncated_normal(shape,stddev=0.1)
       return tf.get_variable(name, initializer = initial)

    if initializer == 'Xavier' :
       initial = tf.contrib.layers.xavier_initializer()
       return tf.get_variable(name, shape, initializer=initial)
    else:
       raise NameError("Invalid Initializer Name with Weight Variable")


def batch_normalization(inputs):
   input_shape = inputs.get_shape()
   parameter_shape = input_shape[-1:]

   axis = list(range(len(input_shape)-1))
   epsilon = 0.001
 
   gamma = weight_variable('bn_gammma',parameter_shape)
   beta = bias_variable('bn_beta',parameter_shape)

   mean, variance = tf.nn.moments(inputs,axis)

   return tf.nn.batch_normalization(inputs,mean,variance,beta,gamma,epsilon)
  

def CNN_module(inputs,operation,filter_in,filter_out):

   with tf.variable_scope('bn1') as scope:
       bn1 = tf.nn.relu(batch_normalization(inputs))

   with tf.variable_scope('conv1') as scope:
       W1 = weight_variable('W1',[3,3,filter_in,filter_out])
       conv1 = tf.nn.conv2d(bn1,W1,strides=[1,1,1,1],padding='SAME')

   with tf.variable_scope('bn2') as scope:   
     bn2 = tf.nn.relu(batch_normalization(conv1))

   with tf.variable_scope('conv2') as scope:
     W2 = weight_variable('W2',[3,3,filter_out,filter_out])
     conv2 = tf.nn.conv2d(bn2,W2,strides=[1,1,1,1],padding='SAME')

   if operation == 'Forward':
     return conv2
   if operation == 'Sum':
     if(inputs.get_shape() == conv2.get_shape()):
       return tf.add(inputs,conv2)
     else:
       raise TypeError("Cannot add : different size")
   if operation == 'Concatnate':
     return tf.concat([inputs,conv2],3)
   

def LRNN_module(inputs,operation1,operation2,operation_module,filter_in,filter_out):

   with tf.variable_scope('bn1') as scope:
     bn1 = tf.nn.relu(batch_normalization(inputs))

   with tf.variable_scope('LRNN_LR') as scope:
     LRNN_LR = Layer_recurrent(bn1,filter_out,cell_type='Basic',direction_to='LR',result_op=operation1)

   with tf.variable_scope('bn2') as scope:
     bn2 = tf.nn.relu(batch_normalization(LRNN_LR))

   with tf.variable_scope('LRNN_UD') as scope:
     LRNN_UD = Layer_recurrent(bn2,filter_out,cell_type='Basic',direction_to='UD',result_op=operation2)
 
   if operation_module == 'Forward':
     return LRNN_UD
   if operation_module == 'Sum':
     if(inputs.get_shape() == LRNN_UD.get_shape()):
       return tf.add(inputs,LRNN_UD)
     else:
       raise TypeError("Cannot add : different size")

   if operation_module == 'Concatnate':
     return tf.concat([inputs,LRNN_UD],3)


  


#Make Bias Variables with 0.1 constant
def bias_variable(name, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.get_variable(name, initializer = initial)

#Make Feedforward hidden layer with Activations
#input = [Number of Data * Number of Input Features], wegiht = [Number of input features * output features]
#bias = [# of output features]
#return = [Number of Data * Number of Output Features]
def full_connected(inputs,weight,bias,activation='ReLU'):
        if activation == 'ReLU':
                return tf.nn.relu(tf.matmul(inputs,weight)+bias, name='relu')
        if activation == 'Sigmoid':
                return tf.nn.sigmoid(tf.matmul(inputs,weight)+bias, name='sigmoid')
        if activation == 'Softmax':
                return tf.nn.softmax(tf.matmul(inputs,weight)+bias, name='softmax')


def loss_function(y_true,y_out,loss_type = 'cross_entropy', regularization = 'L2', REGULARIZATION_PARA=0.01):
        #Chosse Loss function Type: Cross_entropy --> classification, MSE --> Regression
        if loss_type == 'cross_entropy':
                epsilon = tf.constant(value=0.00001)
                loss = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_out+epsilon),axis=1),axis=0)
        if loss_type == 'MSE':
                loss = tf.reduce_mean(tf.square(y_out-y_true))
        if loss_type == 'TimeSeries_MSE':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_out-y_true),axis=2))

        if REGULARIZATION == True:
                #Choose Regularizaton Type: L1, L2 
                vars = tf.trainable_variables()
                if regularization == 'L2':
                        loss = loss + tf.add_n([ tf.nn.l2_loss(v) for v in vars ]) * REGULARIZATION_PARA
                if regularization == 'L1':
                        loss = loss + tf.add_n([ tf.nn.l1_loss(v) for v in vars ]) * REGULARIZATION_PARA

        return loss


def classification_accuracy(test_out, y_test):
	one_hot_result = np.argmax(test_out,axis=1)
	one_hot_true = np.argmax(y_test, axis=1)
	test_size = len(y_test)
	temp_count = 0
	for i in range(test_size):
		if one_hot_result[i] == one_hot_true[i]:
			temp_count +=1

	accuracy = float(temp_count)/test_size
	print accuracy
	return accuracy