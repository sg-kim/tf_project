from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import time

import tensorflow as tf
sess = tf.Session()

cudnn_on_gpu = False

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], use_cudnn_on_gpu=cudnn_on_gpu, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

#######################################################################
##          Input layer
#######################################################################

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#######################################################################
##          First convolution and pooling layers
#######################################################################

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#######################################################################
##          Second convolution and pooling layers
#######################################################################

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#######################################################################
##          Fully connected neuron layer
#######################################################################

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#######################################################################
##          Dropout operation
#######################################################################

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#######################################################################
##          Output layer
#######################################################################

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

z = tf.nn.softmax(y_conv)

#######################################################################
##          Training
#######################################################################

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

start_time = time.time()

#for i in range(20000):
for i in range(5000):    
    batch = mnist.train.next_batch(50)
    if i%20 == 0:
        #train_accuracy = sess.run(accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %3f"%(i, train_accuracy))
    #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

test = mnist.test.next_batch(1000)

training_time = time.time()

test_accuracy = sess.run(accuracy, feed_dict={x: test[0], y_: test[1], keep_prob: 1.0})
print("test accuracy %f"%test_accuracy)

end_time = time.time()
print("Run time %f"%(end_time - start_time))
print("Training time %f"%(training_time - start_time))
print("Test time %f"%(end_time - training_time))

