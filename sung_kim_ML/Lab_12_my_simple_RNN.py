import tensorflow as tf
import rnn_cell_sgkim as rcell


x = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 4])

ht = tf.Variable([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)

##Wxh = tf.Variable(tf.random_normal([4, 4]), name='Weight_xh')
##Whh = tf.Variable(tf.random_normal([4, 4]), name='Weight_hh')

Wxh = tf.Variable([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0],
                   [5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0]], name='Weight_xh')
Whh = tf.Variable([[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0],
                   [6.0, 6.0, 6.0, 6.0], [8.0, 8.0, 8.0, 8.0]], name='Weight_hh')

##ht = tf.nn.tanh(tf.matmul(ht, Whh) + tf.matmul(x, Wxh))
fb_loop = tf.assign(ht, tf.matmul(ht, Whh) + tf.matmul(x, Wxh))

##Wyh = tf.Variable(tf.random_normal([4, 4], name='Weight_yh'))
Wyh = tf.Variable([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0],
                   [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]], name='Weight_yh')

hypo = tf.matmul(fb_loop, Wyh)

##hypo2 = rcell.rnn_cell(2, 4, 4, x)
hypo2 = rcell.rnn_cell(2, 4, 4)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1, 4):

    ht_val, hypo_val = sess.run([ht, hypo], feed_dict={x:[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]})

##    print(ht_val)
    print(hypo_val)

    hypo2_val = hypo2.run(sess, [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    print(hypo2_val)

