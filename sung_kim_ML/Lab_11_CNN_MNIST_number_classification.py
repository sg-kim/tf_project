import tensorflow as tf
import math
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def xavier_init(n_inputs, n_outputs, uniform=True):
	"""Set the parameter initialization using the method described.
	This method is designed to keep the scale of the gradients roughly the same
	in all layers.
	Xavier Glorot and Yoshua Bengio (2010):
		Understanding the difficulty of training deep feedforward neural
		networks. International conference on artificial intelligence and
		statistics.
	Args:
		n_inputs: The number of input nodes into each output.
		n_outputs: The number of output nodes for each input.
		uniform: If true use a uniform distribution, otherwise use a normal.
	Returns:
	An initializer.
	"""
	if uniform:
		# 6 was used in the paper.
		init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
		return tf.random_uniform_initializer(-init_range, init_range)
	else:
		# 3 gives us approximately the same limits as above since this repicks
		# values greater than 2 standard deviations from the mean.
		stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
		return tf.truncated_normal_initializer(stddev=stddev)

x = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

x_for_conv = tf.reshape(x, [-1, 28, 28, 1], name='input_reshape')

cudnn_on_gpu = True

with tf.name_scope("conv_7x7_layer_1") as scope:
    #W_conv_l1 = tf.Variable(tf.random_uniform([7, 7, 1, 64], -1.0, 1.0))
    #b_conv_l1 = tf.Variable(tf.random_uniform([1, 64], -1.0, 1.0))
    W_conv_l1 = tf.get_variable("W_conv_l1", shape=[7, 7, 1, 64], initializer=xavier_init(49, 3136))
    b_conv_l1 = tf.get_variable("b_conv_l1", shape=[1, 64], initializer=xavier_init(64, 64))

    conv_l1 = tf.nn.relu(tf.nn.conv2d(x_for_conv, W_conv_l1, strides=[1, 1, 1, 1],
                           use_cudnn_on_gpu=cudnn_on_gpu, padding='SAME') + b_conv_l1,
                         name='conv_7x7_layer_1')

with tf.name_scope("conv_5x5_layer_2") as scope:
    #W_conv_l2 = tf.Variable(tf.random_uniform([5, 5, 64, 64], -1.0, 1.0))
    #b_conv_l2 = tf.Variable(tf.random_uniform([1, 64], -1.0, 1.0))
    W_conv_l2 = tf.get_variable("W_conv_l2", shape=[5, 5, 64, 64], initializer=xavier_init(1600, 1600))
    b_conv_l2 = tf.get_variable("b_conv_l2", shape=[1, 64], initializer=xavier_init(64, 64))


    conv_l2 = tf.nn.relu(tf.nn.conv2d(conv_l1, W_conv_l2, strides=[1, 1, 1, 1],
                           use_cudnn_on_gpu=cudnn_on_gpu, padding='SAME') + b_conv_l2,
                         name='conv_5x5_layer_2')

with tf.name_scope("max_pooling_2x2_layer_3") as scope:
    mxp_l3 = tf.nn.max_pool(conv_l2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            name='max_pool_2x2_layer_3')

with tf.name_scope("conv_7x7_layer_4") as scope:
    #W_conv_l4 = tf.Variable(tf.random_uniform([7, 7, 64, 32], -1.0, 1.0))
    #b_conv_l4 = tf.Variable(tf.random_uniform([1, 32], -1.0, 1.0))
    W_conv_l4 = tf.get_variable("W_conv_l4", shape=[7, 7, 64, 32], initializer=xavier_init(3136, 1568))
    b_conv_l4 = tf.get_variable("b_conv_l4", shape=[1, 32], initializer=xavier_init(32, 32))

    conv_l4 = tf.nn.relu(tf.nn.conv2d(mxp_l3, W_conv_l4, strides=[1, 1, 1, 1],
                           use_cudnn_on_gpu=cudnn_on_gpu, padding='SAME') + b_conv_l4,
                         name='conv_7x7_layer_4')
    
with tf.name_scope("conv_5x5_layer_5") as scope:
    #W_conv_l5 = tf.Variable(tf.random_uniform([5, 5, 32, 32], -1.0, 1.0))
    #b_conv_l5 = tf.Variable(tf.random_uniform([1, 32], -1.0, 1.0))
    W_conv_l5 = tf.get_variable("W_conv_l5", shape=[5, 5, 32, 32], initializer=xavier_init(800, 800))
    b_conv_l5 = tf.get_variable("b_conv_l5", shape=[1, 32], initializer=xavier_init(32, 32))

    conv_l5 = tf.nn.relu(tf.nn.conv2d(conv_l4, W_conv_l5, strides=[1, 1, 1, 1],
                           use_cudnn_on_gpu=cudnn_on_gpu, padding='SAME') + b_conv_l5,
                         name='conv_5x5_layer_5')

with tf.name_scope("max_pooling_2x2_layer_6") as scope:
    mxp_l6 = tf.nn.max_pool(conv_l5, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            name='max_pool_2x2_layer_6')

mxp_l6_for_fc = tf.reshape(mxp_l6, [-1, 1568])
prob_keep = tf.placeholder(tf.float32)

with tf.name_scope("full_connection_layer_7") as scope:
    #W_fc_l7 = tf.Variable(tf.random_uniform([1568, 1024], -1.0, 1.0))
    #b_fc_l7 = tf.Variable(tf.random_uniform([1, 1024], -1.0, 1.0))
    W_fc_l7 = tf.get_variable("W_fc_l7", shape=[1568, 1024], initializer=xavier_init(1568, 1024))
    b_fc_l7 = tf.get_variable("b_fc_l7", shape=[1, 1024], initializer=xavier_init(1024, 1024))

    fc_l7 = tf.nn.relu(tf.matmul(mxp_l6_for_fc, W_fc_l7) + b_fc_l7, name='fully_conn_layer_7')
    fc_l7 = tf.nn.dropout(fc_l7, prob_keep, name='dropout_fc_layer7')

with tf.name_scope("full_connection_layer_8") as scope:
    #W_fc_l8 = tf.Variable(tf.random_uniform([1024, 256], -1.0, 1.0))
    #b_fc_l8 = tf.Variable(tf.random_uniform([1, 256], -1.0, 1.0))
    W_fc_l8 = tf.get_variable("W_fc_l8", shape=[1024, 256], initializer=xavier_init(1024, 256))
    b_fc_l8 = tf.get_variable("b_fc_l8", shape=[1, 256], initializer=xavier_init(256, 256))

    fc_l8 = tf.nn.relu(tf.matmul(fc_l7, W_fc_l8) + b_fc_l8, name='fully_conn_layer_8')
    fc_l8 = tf.nn.dropout(fc_l8, prob_keep, name='dropout_fc_layer8')

with tf.name_scope("full_connection_layer_9") as scope:
    #W_fc_l9 = tf.Variable(tf.random_uniform([256, 10], -1.0, 1.0))
    #b_fc_l9 = tf.Variable(tf.random_uniform([1, 10], -1.0, 1.0))
    W_fc_l9 = tf.get_variable("W_fc_l9", shape=[256, 10], initializer=xavier_init(256, 10))
    b_fc_l9 = tf.get_variable("b_fc_l9", shape=[1, 10], initializer=xavier_init(10, 10))
    
    fc_l9 = tf.matmul(fc_l8, W_fc_l9) + b_fc_l9
    hypo = tf.nn.softmax(fc_l9, name='hypothesis-fc_layer_9')

with tf.name_scope("cost_function") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=fc_l9, name='cost'))

    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

validation = tf.equal(tf.arg_max(hypo, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(validation, tf.float32))

acc_summ = tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/Lab_11_CNN_MNIST_deeper", sess.graph)

start_time = time.time()

for step in range(0, 2001):

    batch = mnist.train.next_batch(50)

    sess.run(optimizer, feed_dict={x:batch[0], Y:batch[1], prob_keep:0.5})

    if step%20 == 0:

        cost_val = sess.run(cost, feed_dict={x:batch[0], Y:batch[1], prob_keep:0.5})
        acc_val, summary = sess.run([accuracy, merged], feed_dict={x:batch[0], Y:batch[1], prob_keep:1.0})

        writer.add_summary(summary, step)

        print("Step: %d, cost: %f, accuracy: %f"%(step, cost_val, acc_val))

end_training = time.time()

test = mnist.test.next_batch(1000)

acc_val = sess.run(accuracy, feed_dict={x:test[0], Y:test[1], prob_keep:1.0})

end_test = time.time()

print("run time: %f"%(end_test - start_time))
print("time for training: %f"%(end_training - start_time))
print("time for test: %f"%(end_test - end_training))
print("accuracy: %f"%(acc_val))
