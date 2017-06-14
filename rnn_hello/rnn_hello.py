import tensorflow as tf
import numpy as np
import time

char_raw_dic = ['h', 'e', 'l', 'o']     #   index - value
char_dic = {w: i for i, w in enumerate(char_raw_dic)}       #   value - index
x_data = np.array([[1, 0, 0, 0],                        #   h
                   [0, 1, 0, 0],                        #   e
                   [0, 0, 1, 0],                        #   l
                   [0, 0, 1, 0]], dtype=np.float32)     #   l

sample = [char_dic[c] for c in 'hello']

#######################################################################
##  Inputs
#######################################################################

i_0 = tf.placeholder(tf.float32, shape=[None, 1])
i_1 = tf.placeholder(tf.float32, shape=[None, 1])
i_2 = tf.placeholder(tf.float32, shape=[None, 1])
i_3 = tf.placeholder(tf.float32, shape=[None, 1])

z_0_ = tf.placeholder(tf.float32, shape=[None, 4])
z_1_ = tf.placeholder(tf.float32, shape=[None, 4])
z_2_ = tf.placeholder(tf.float32, shape=[None, 4])
z_3_ = tf.placeholder(tf.float32, shape=[None, 4])

#######################################################################
##  Shared weights
#######################################################################

Wi = tf.Variable(tf.random_normal([1, 32]))
bi = tf.Variable(tf.random_normal([32]))

Wf = tf.Variable(tf.random_normal([32, 32]))
bf = tf.Variable(tf.random_normal([32]))

Wz = tf.Variable(tf.random_normal([32, 4]))
bz = tf.Variable(tf.random_normal([4]))

#######################################################################
##  BPTT t0
#######################################################################

y_0 = tf.nn.sigmoid(tf.matmul(i_0, Wi) + bi)        ##  input gate

z_0 = tf.nn.softmax(tf.matmul(y_0, Wz) + bz)        ##  output gate

#######################################################################
##  BPTT t1
#######################################################################

y_1 = tf.nn.sigmoid(tf.matmul(i_1, Wi) + bi) + tf.nn.sigmoid((y_0, Wf) + bf)        ##  input gate & output gate

z_1 = tf.nn.softmax(tf.matmul(y_1, Wz) + bz)

#######################################################################
##  BPTT t2
#######################################################################

y_2 = tf.nn.sigmoid(tf.matmul(i_2, Wi) + bi) + tf.nn.sigmoid((y_1, Wf) + bf)

z_2 = tf.nn.softmax(tf.matmul(y_2, Wz) + bz)

#######################################################################
#   BPTT t3
#######################################################################

y_3 = tf.nn.sigmoid(tf.matmul(i_3, Wi) + bi) + tf.nn.sigmoid((y_2, Wf) + bf)

z_3 = tf.nn.softmax(tf.matmul(y_3, Wz) + bz)

#######################################################################
#   Training
#######################################################################

x_entropy_z_0 = tf.nn.softmax_cross_entropy_with_logits(labels=x_data[0], z_0)
x_entropy_z_1 = tf.nn.softmax_cross_entropy_with_logits(labels=x_data[1], z_1)
x_entropy_z_2 = tf.nn.softmax_cross_entropy_with_logits(labels=x_data[2], z_2)
x_entropy_z_3 = tf.nn.softmax_cross_entropy_with_logits(labels=x_data[3], z_3)


