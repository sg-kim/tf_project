import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]

y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]


x = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

h = tf.matmul(x, W) + b
hypo = tf.nn.softmax(h)

cost = -tf.reduce_mean(tf.reduce_sum(Y*tf.log(hypo), axis=1))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(0, 1000):

    sess.run(optimizer, feed_dict={x:x_data, Y:y_data})

    if step%10 == 0:

        print(sess.run(cost, feed_dict={x:x_data, Y:y_data}))


test_val = sess.run(hypo, feed_dict={x:[[1, 2, 3, 4]], Y:[[0, 0, 1]]})

print(test_val, sess.run(tf.arg_max(test_val, 1)))
