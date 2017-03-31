import tensorflow as tf`

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

############################################################################
##      Logistic regression model
############################################################################

h = tf.matmul(x, W) + b
hypo = tf.divide(1, (1 + tf.exp(-h)))

############################################################################
##      Optimizer definitions
############################################################################

cost = -tf.reduce_mean(Y*tf.log(hypo) + (1 - Y)*tf.log(1 - hypo))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

prediction = tf.cast(hypo > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

############################################################################
##      Training
############################################################################

for step in range(0, 2001):

    sess.run(optimizer, feed_dict = {x:x_data, Y:y_data})

    if step%10 == 0:

        cost_val = sess.run(cost, feed_dict = {x:x_data, Y:y_data})
        print('step: %d cost: %.3f'%(step, cost_val))

############################################################################
##      Evaluation
############################################################################

print(sess.run([hypo, prediction], feed_dict = {x:[[7, 3]]}))
print(sess.run([hypo, prediction], feed_dict = {x:[[3, 3]]}))
print(sess.run([hypo, prediction], feed_dict = {x:[[5, 1]]}))

print(sess.run(accuracy, feed_dict = {x:[[7, 3], [3, 3], [5, 1]], Y:[[1], [0], [0]]}))
