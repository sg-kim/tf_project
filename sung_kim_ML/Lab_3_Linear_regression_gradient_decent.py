import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypo = W*X

cost = tf.reduce_mean(tf.square(hypo - Y))

learning_rate = 0.1
gradient = tf.reduce_mean(W*X - Y)
decent = W - learning_rate*gradient
update = W.assign(decent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):

    sess.run(update, feed_dict={X:[1, 2, 3], Y:[1, 2, 3]})

    if step%20 == 0:
        print(step, sess.run(cost, feed_dict={X:[1, 2, 3], Y:[1, 2, 3]}), sess.run(W))
