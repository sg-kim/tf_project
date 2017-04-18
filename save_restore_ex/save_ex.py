import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 2]), name="weight_1")
w2 = tf.Variable(tf.random_normal([3, 3]), name="weight_2")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

w1_val, w2_val = sess.run([w1, w2])

print(w1_val)
print(w2_val)

saver = tf.train.Saver({"weight_1":w1, "weight_2":w2})
save_path = saver.save(sess, "./model.wgt")

print("save path: %s"%save_path)
