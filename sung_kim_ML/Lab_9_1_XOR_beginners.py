import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

x_test = [[1, 0], [1, 1], [0, 0], [0, 1]]
y_test = [[1], [0], [0], [1]]


x = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])


W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([1, 2]), name='bias1')

y1 = tf.matmul(x, W1) + b1
h1 = tf.nn.sigmoid(y1)

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1, 1]), name='bias2')

y2 = tf.matmul(h1, W2) + b2
h2 = tf.nn.sigmoid(y2)

cost = -tf.reduce_mean(Y*tf.log(h2) + (1 - Y)*tf.log(1 - h2))

optimize = tf.train.GradientDescentOptimizer(1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(0, 1001):

    sess.run(optimize, feed_dict={x:x_data, Y:y_data})

    if step%10 == 0:

        cost_val = sess.run(cost, feed_dict={x:x_data, Y:y_data})

        print(cost_val)

prediction = sess.run(tf.cast(h2 > 0.5, tf.float32), feed_dict={x:x_test})
W1_val, b1_val, W2_val, b2_val = sess.run([W1, b1, W2, b2])

print(W1_val)
print(b1_val)
print(W2_val)
print(b2_val)
print(prediction)
