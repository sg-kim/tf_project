import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

x_test = [[1, 0], [1, 1], [0, 0], [0, 1]]
y_test = [[1], [0], [0], [1]]


x = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

with tf.name_scope("layer1") as scope:

    W1 = tf.Variable(tf.random_normal([2, 4]), name='weight1')
    b1 = tf.Variable(tf.random_normal([1, 4]), name='bias1')

    y1 = tf.matmul(x, W1) + b1
    h1 = tf.nn.sigmoid(y1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("bias1", b1)

with tf.name_scope("layer2") as scope:

    W2 = tf.Variable(tf.random_normal([4, 2]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1, 2]), name='bias2')

    y2 = tf.matmul(h1, W2) + b2
    h2 = tf.nn.sigmoid(y2)

    w2_hist = tf.summary.histogram("weights1", W2)
    b2_hist = tf.summary.histogram("bias1", b2)

with tf.name_scope("layer3") as scope:

    W3 = tf.Variable(tf.random_normal([2, 1]), name='weight3')
    b3 = tf.Variable(tf.random_normal([1, 1]), name='bias3')

    y3 = tf.matmul(h2, W3) + b3
    h3 = tf.nn.sigmoid(y3)

    w3_hist = tf.summary.histogram("weights1", W3)
    b3_hist = tf.summary.histogram("bias1", b3)

with tf.name_scope("cost") as scope:

    cost = -tf.reduce_mean(Y*tf.log(h3) + (1 - Y)*tf.log(1 - h3))

    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("optimize") as scope:
    
    optimize = tf.train.GradientDescentOptimizer(1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/xor_deep_logs", sess.graph)

for step in range(0, 1001):

    sess.run(optimize, feed_dict={x:x_data, Y:y_data})

    if step%10 == 0:

        summary, cost_val = sess.run([merged, cost], feed_dict={x:x_data, Y:y_data})
        writer.add_summary(summary, step)

        print(cost_val)

prediction = sess.run(tf.cast(h3 > 0.5, tf.float32), feed_dict={x:x_test})
W1_val, b1_val, W2_val, b2_val, W3_val, b3_val = sess.run([W1, b1, W2, b2, W3, b3])

print(W1_val)
print(b1_val)
print(W2_val)
print(b2_val)
print(W3_val)
print(b3_val)
print(prediction)
