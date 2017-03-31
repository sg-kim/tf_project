import tensorflow as tf

###############################################################
##      Train data
###############################################################
#x_train = [1, 2, 3]
#y_train = [1, 2, 3]

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

###############################################################
##      Linear regression model
###############################################################
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypo = W*X + b

###############################################################
##      Cost function
###############################################################
cost = tf.reduce_mean(tf.square(hypo - Y))

###############################################################
##      Optimizer
###############################################################
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

###############################################################
##      Training
###############################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):    #   0 ~ 2000
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1, 2, 3], Y:[1, 2, 3]})

    if step%20 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(hypo, feed_dict={X: [5]}))
print(sess.run(hypo, feed_dict={X: [2.5]}))
