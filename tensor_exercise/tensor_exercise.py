import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape = shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial  = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

W_conv1 = weight_variable([5, 5, 1, 3])
b_conv1 = bias_variable([32])
x_conv1 = weight_variable([3, 3])

w = weight_variable([3, 2, 3])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W = sess.run(W_conv1)
b = sess.run(b_conv1)
x = sess.run(x_conv1)

x_image = sess.run(tf.reshape(x, [-1, 3, 3, 1]))

import numpy as np

#t = np.array([[[111, 112, 113], [121, 122, 123]], [[211, 212, 213], [221, 222, 223]], [[311, 312, 313], [321, 322, 323]]])
t = np.array([[[111, 112, 113], [121, 122, 123], [131, 132, 133]], [[211, 212, 213], [221, 222, 223], [231, 232, 233]]])
w = sess.run(w)

t2 = np.array([[[[1111, 1112, 1113], [1121, 1122, 1123], [1131, 1132, 1133]], [[1211, 1212, 1213], [1221, 1222, 1223], [1231, 1232, 1233]]], [[[2111, 2112, 2113], [2121, 2122, 2123], [2131, 2132, 2133]], [[2211, 2212, 2213], [2221, 2222, 2223], [2231, 2232, 2233]]]])

t3 = sess.run(tf.reshape(t2, [-1, 6, 3]))
t4 = sess.run(tf.reshape(t, [-1, 1, 2, 3]))

