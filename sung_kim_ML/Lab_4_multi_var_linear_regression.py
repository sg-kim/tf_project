import tensorflow as tf
import numpy as np

####################################################################
##      Direct input
####################################################################

#x_train = np.array([[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]])
#y_train = np.array([[152], [185], [180], [196], [142]])

####################################################################
##      Input from file using numpy method
####################################################################

#xy = np.loadtxt('lab_4_data_01_test_score.csv', delimiter=',', dtype=np.float32)

#x_train = xy[:, 0:-1]
#y_train = xy[:, [-1]]

#print(x_train)
#print(y_train)

####################################################################
##      Input from file using tensorflow queue runners
####################################################################

filename_queue = tf.train.string_input_producer(['Lab_4_data_01_test_score.csv'],
                                                shuffle=False,
                                                name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=5)

####################################################################
##      Multi variable linear regression hypothesis
####################################################################

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal(shape=[3, 1]), name='weight')
b = tf.Variable(tf.random_normal(shape=[1]), name='bias')

hypo = tf.matmul(x, W) + b

####################################################################
##      Cost and optimization method
####################################################################

cost = tf.reduce_mean(tf.square(hypo - y))

optimize = tf.train.GradientDescentOptimizer(1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

####################################################################
##      Training
####################################################################

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):

    x_train = sess.run(train_x_batch)
    y_train = sess.run(train_y_batch)

    sess.run(optimize, feed_dict = {x:x_train, y:y_train})

    if step%20 == 0:

        cost_val, W_val, b_val = sess.run([cost, W, b], feed_dict = {x:x_train, y:y_train})
        
        #print(sess.run(cost, feed_dict = {x:x_train, y:y_train}), sess.run(W), sess.run(b))

        print(cost_val, W_val[0], W_val[1], W_val[2], b_val)


coord.request_stop()
coord.join(threads)

####################################################################
##      Evaluation
####################################################################

print(sess.run(hypo, feed_dict={x:[[100, 70, 101]]}))
print(sess.run(hypo, feed_dict={x:[[60, 70, 110], [90, 100, 80]]}))

    
