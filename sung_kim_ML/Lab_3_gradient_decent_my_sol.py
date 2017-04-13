import tensorflow as tf
#import matplotlib.pyplot as plt


x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')

cost = tf.reduce_mean(tf.square(W*x - y))
gradient = tf.reduce_mean((W*x - y)*x)
learning_rate = 0.01
decent = learning_rate*gradient
update = W.assign(W - decent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):

    sess.run(update, feed_dict={x:[1, 2, 3], y:[1, 2, 3]})

    if step%10 == 0:
        print(sess.run(cost, feed_dict={x:[1, 2, 3], y:[1, 2, 3]}), sess.run(W)) 

