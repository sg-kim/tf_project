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

nb_classes = 4
sample_one_hot = tf.one_hot(sample, nb_classes)

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

y_wb_0 = tf.matmul(y_0, Wz) + bz

z_0 = tf.nn.softmax(y_wb_0)        ##  output gate

#######################################################################
##  BPTT t1
#######################################################################

y_1 = tf.nn.sigmoid(tf.matmul(i_1, Wi) + bi) + tf.nn.sigmoid(tf.matmul(y_0, Wf) + bf)        ##  input gate & output gate

y_wb_1 = tf.matmul(y_1, Wz) + bz

z_1 = tf.nn.softmax(y_wb_1)

#######################################################################
##  BPTT t2
#######################################################################

y_2 = tf.nn.sigmoid(tf.matmul(i_2, Wi) + bi) + tf.nn.sigmoid(tf.matmul(y_1, Wf) + bf)

y_wb_2 = tf.matmul(y_2, Wz) + bz

z_2 = tf.nn.softmax(y_wb_2)

#######################################################################
#   BPTT t3
#######################################################################

y_3 = tf.nn.sigmoid(tf.matmul(i_3, Wi) + bi) + tf.nn.sigmoid(tf.matmul(y_2, Wf) + bf)

y_wb_3 = tf.matmul(y_3, Wz) + bz

z_3 = tf.nn.softmax(y_wb_3)

#######################################################################
#   Training
#######################################################################

x_entropy_z_0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x_data[0], y_wb_0))
x_entropy_z_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x_data[1], y_wb_1))
x_entropy_z_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x_data[2], y_wb_2))
x_entropy_z_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x_data[3], y_wb_3))

cross_entropy = tf.reduce_mean([x_entropy_z_0, x_entropy_z_1, x_entropy_z_2, x_entropy_z_3])

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = [tf.equal(tf.argmax(z_0, 1), sample[1]),
                    tf.equal(tf.argmax(z_1, 1), sample[2]),
                    tf.equal(tf.argmax(z_2, 1), sample[3]),
                    tf.equal(tf.argmax(z_3, 1), sample[4])]

accuracy = tf.reduce_mean(correct_prediction)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={i_0: x_data[0], i_1: x_data[1], i_2: x_data[2], i_3: x_data[3],
                                    z_0_: sample_one_hot[1], z_1_: sample_one_hot[1], z_2_: sample_one_hot[1], z_3_: sample_one_hot[1]})
    
test_accuracy = sess.run(correct_prediction)/4.0

print("test accuracy %f"%test_accuracy)


