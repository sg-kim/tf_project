from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

#########################################################
# Building the model to identify MNIST digit data
#########################################################

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b
z = tf.nn.softmax(y)

y_ = tf.placeholder(tf.float32, [None, 10])

#########################################################
# Training the model
#########################################################

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), 1))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = [y_], logits = [y]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#########################################################
# Evaluating the model
#########################################################

#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
