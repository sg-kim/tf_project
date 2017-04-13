import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

x_test = [[1, 0], [1, 1], [0, 0], [0, 1]]
y_test = [[1], [0], [0], [1]]

x = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

with tf.name_scope("layer1") as scope:

    W1 = tf.Variable(tf.random_uniform([2, 6], -1.0, 1.0))
    b1 = tf.Variable(tf.random_uniform([1, 6], -1.0, 1.0))
    h1 = tf.sigmoid(tf.matmul(x, W1) + b1)

with tf.name_scope("layer2") as scope:
    
    W2 = tf.Variable(tf.random_uniform([6, 8], -1.0, 1.0))
    b2 = tf.Variable(tf.random_uniform([1, 8], -1.0, 1.0))
    h2 = tf.sigmoid(tf.matmul(h1, W2) + b2)

with tf.name_scope("layer3") as scope:

    W3 = tf.Variable(tf.random_uniform([8, 8], -1.0, 1.0))
    b3 = tf.Variable(tf.random_uniform([1, 8], -1.0, 1.0))
    h3 = tf.sigmoid(tf.matmul(h2, W3) + b3)

with tf.name_scope("layer4") as scope:
    
    W4 = tf.Variable(tf.random_uniform([8, 12], -1.0, 1.0))
    b4 = tf.Variable(tf.random_uniform([1, 12], -1.0, 1.0))
    h4 = tf.sigmoid(tf.matmul(h3, W4) + b4)

with tf.name_scope("layer5") as scope:
    
    W5 = tf.Variable(tf.random_uniform([12, 10], -1.0, 1.0))
    b5 = tf.Variable(tf.random_uniform([1, 10], -1.0, 1.0))
    h5 = tf.sigmoid(tf.matmul(h4, W5) + b5)

with tf.name_scope("layer6") as scope:
    
    W6 = tf.Variable(tf.random_uniform([10, 6], -1.0, 1.0))
    b6 = tf.Variable(tf.random_uniform([1, 6], -1.0, 1.0))
    h6 = tf.sigmoid(tf.matmul(h5, W6) + b6)

with tf.name_scope("layer7") as scope:

    W7 = tf.Variable(tf.random_uniform([6, 4], -1.0, 1.0))
    b7 = tf.Variable(tf.random_uniform([1, 4], -1.0, 1.0))
    h7 = tf.sigmoid(tf.matmul(h6, W7) + b7)

with tf.name_scope("layer8") as scope:
    
    W8 = tf.Variable(tf.random_uniform([4, 2], -1.0, 1.0))
    b8 = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    h8 = tf.sigmoid(tf.matmul(h7, W8) + b8)

with tf.name_scope("layer9") as scope:
    
    W9 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
    b9 = tf.Variable(tf.random_uniform([1, 1], -1.0, 1.0))
    h9 = tf.sigmoid(tf.matmul(h8, W9) + b9)

with tf.name_scope("cost") as scope:
    
    cost = tf.reduce_mean(-(Y*tf.log(h9) + (1 - Y)*tf.log(1 - h9)))

    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("optimizer") as scope:
    
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.name_scope("evaluation") as scope:
    
    prediction = tf.cast(h9 > 0.5, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32))

    acc_summ = tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/xor_deeper_w_sigmoid", sess.graph)

for step in range(0, 1001):

    sess.run(optimizer, feed_dict={x:x_data, Y:y_data})

    if step%10 == 0:

        cost_val, acc_val, summary = sess.run([cost, accuracy, merged], feed_dict={x:x_data, Y:y_data})

        writer.add_summary(summary, step)

        print(cost_val)

print(sess.run(accuracy, feed_dict={x:x_test, Y:y_test}))
