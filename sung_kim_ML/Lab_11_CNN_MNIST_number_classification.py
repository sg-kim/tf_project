import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

x_for_conv = tf.reshape(x, [-1, 28, 28, 1], name='input_reshape')

with tf.name_scope("conv_7x7_layer_1") as scope:
    W_conv_l1 = tf.Variable(tf.random_uniform([7, 7, 1, 64], -1.0, 1.0))
    b_conv_l1 = tf.Variable(tf.random_uniform([1, 64], -1.0, 1.0))

    conv_l1 = tf.nn.relu(tf.nn.conv2d(x_for_conv, W_conv_l1, strides=[1, 1, 1, 1],
                           use_cudnn_on_gpu=False, padding='SAME') + b_conv_l1,
                         name='conv_7x7_layer_1')

with tf.name_scope("conv_5x5_layer_2") as scope:
    W_conv_l2 = tf.Variable(tf.random_uniform([5, 5, 64, 64], -1.0, 1.0))
    b_conv_l2 = tf.Variable(tf.random_uniform([1, 64], -1.0, 1.0))

    conv_l2 = tf.nn.relu(tf.nn.conv2d(conv_l1, W_conv_l2, strides=[1, 1, 1, 1],
                           use_cudnn_on_gpu=False, padding='SAME') + b_conv_l2,
                         name='conv_5x5_layer_2')

with tf.name_scope("max_pooling_2x2_layer_3") as scope:
    mxp_l3 = tf.nn.max_pool(conv_l2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            name='max_pool_2x2_layer_3')

with tf.name_scope("conv_7x7_layer_4") as scope:
    W_conv_l4 = tf.Variable(tf.random_uniform([7, 7, 64, 32], -1.0, 1.0))
    b_conv_l4 = tf.Variable(tf.random_uniform([1, 32], -1.0, 1.0))

    conv_l4 = tf.nn.relu(tf.nn.conv2d(mxp_l3, W_conv_l4, strides=[1, 1, 1, 1],
                           use_cudnn_on_gpu=False, padding='SAME') + b_conv_l4,
                         name='conv_7x7_layer_4')
    
with tf.name_scope("conv_5x5_layer_5") as scope:
    W_conv_l5 = tf.Variable(tf.random_uniform([5, 5, 32, 32], -1.0, 1.0))
    b_conv_l5 = tf.Variable(tf.random_uniform([1, 32], -1.0, 1.0))

    conv_l5 = tf.nn.relu(tf.nn.conv2d(conv_l4, W_conv_l5, strides=[1, 1, 1, 1],
                           use_cudnn_on_gpu=False, padding='SAME') + b_conv_l5,
                         name='conv_5x5_layer_5')

with tf.name_scope("max_pooling_2x2_layer_6") as scope:
    mxp_l6 = tf.nn.max_pool(conv_l5, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME',
                            name='max_pool_2x2_layer_6')

mxp_l6_for_fc = tf.reshape(mxp_l6, [-1, 1568])
prob_keep = tf.placeholder(tf.float32)

with tf.name_scope("full_connection_layer_7") as scope:
    W_fc_l7 = tf.Variable(tf.random_uniform([1568, 1024], -1.0, 1.0))
    b_fc_l7 = tf.Variable(tf.random_uniform([1, 1024], -1.0, 1.0))

    fc_l7 = tf.nn.relu(tf.matmul(mxp_l6_for_fc, W_fc_l7) + b_fc_l7, name='fully_conn_layer_7')
    fc_l7 = tf.nn.dropout(fc_l7, prob_keep, name='dropout_fc_layer7')

with tf.name_scope("full_connection_layer_8") as scope:
    W_fc_l8 = tf.Variable(tf.random_uniform([1024, 256], -1.0, 1.0))
    b_fc_l8 = tf.Variable(tf.random_uniform([1, 256], -1.0, 1.0))

    fc_l8 = tf.nn.relu(tf.matmul(fc_l7, W_fc_l8) + b_fc_l8, name='fully_conn_layer_8')
    fc_l8 = tf.nn.dropout(fc_l8, prob_keep, name='dropout_fc_layer8')

with tf.name_scope("full_connection_layer_9") as scope:
    W_fc_l9 = tf.Variable(tf.random_uniform([256, 10], -1.0, 1.0))
    b_fc_l9 = tf.Variable(tf.random_uniform([1, 10], -1.0, 1.0))

    fc_l9 = tf.matmul(fc_l8, W_fc_l9) + b_fc_l9
    hypo = tf.nn.softmax(fc_l9, name='hypothesis-fc_layer_9')

with tf.name_scope("cost_function") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=fc_l9, name='cost'))

with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

validation = tf.equal(tf.arg_max(hypo, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(validation, tf.float32))

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

#for step in range(0, 101):

#    batch = mnist.train.next_batch(50)

#    sess.run(optimizer, feed_dict={x:batch[0], Y:batch[1], prob_keep:0.5})

#    if step%20 = 0:

#        cost_val = sess.run(cost, feed_dict={x:batch[0], Y:batch[1], prob_keep:0.5})
#        acc_val = sess.run(accuracy, feed_dict={x:batch[0], Y:batch[1], prob_keep:1.0})
#        print("%f, %f"%(cost_val, acc_val))

#test = mnist.test.next_batch(1000)

#acc_val = sess.run(accuracy, feed_dict={x:test[0], Y:test[1], prob_keep:1.0})

#print(acc_val)
