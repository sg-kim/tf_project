import tensorflow as tf
import numpy as np
import time

char_raw_dic = ['h', 'e', 'l', 'o']     #   index - value
char_dic = {w: i for i, w in enumerate(char_raw_dic)}       #   value - index
x_data = np.array([[1, 0, 0, 0],                        #   h
                   [0, 1, 0, 0],                        #   e
                   [0, 0, 1, 0],                        #   l
                   [0, 0, 1, 0]], dtype=np.float32)     #   l

sample = [char_dic[c] for c in 'hello']

Wi = tf.Variable(tf.random_normal([4, 32]))
i = tf.placeholder(tf.float32, shape=[None, 4])
bi = tf.Variable(tf.random_normal([32]))

input_gate = tf.nn.sigmoid(tf.matmul(i, Wi) + bi)

Wf = tf.Variable(tf.random_normal([32, 32]))
y = tf.Variable(tf.zeros([32, 32]))
bf = tf.Variable(tf.random_normal([32]))

forget_gate = tf.nn.sigmoid(tf.matmul(y, Wf) + bf)

update_y = tf.assign(y, forget_gate)


