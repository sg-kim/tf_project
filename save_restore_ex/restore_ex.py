import tensorflow as tf

w1 = tf.Variable(tf.random_normal([3, 3]), name="weight_1")
w2 = tf.Variable(tf.random_normal([2, 2]), name="weight_2")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver({"weight_1":w2, "weight_2":w1})
saver.restore(sess, "./model.wgt")

w1_val, w2_val = sess.run([w1, w2])

print(w1_val)
print(w2_val)
