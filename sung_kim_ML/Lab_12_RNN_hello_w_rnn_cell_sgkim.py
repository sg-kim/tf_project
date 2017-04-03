import tensorflow as tf
import numpy as np

from rnn_cell_sgkim import rnn_cell

char_rdic = ['h', 'e', 'l', 'o']        #   idex - value
char_dic = {w: i for i, w in enumerate(char_rdic)}      #   value - index
x_data = np.array([[1, 0, 0, 0],                    #   h
                   [0, 1, 0, 0],                    #   e
                   [0, 0, 1, 0],                    #   l
                   [0, 0, 0, 1]], dtype=float)      #   o

sample = [char_dic[c] for c in 'hello']

x = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 4])

char_vocab_size = len(char_dic)
rnn_size = char_vocab_size
time_step_size = 4

rnn_cell = rnn_cell(time_step_size, 4, 4, x)
hypo = tf.one_hot(tf.arg_max(rnn_cell, 1), char_vocab_size)

cost = tf.reduce_mean(tf.reduce_sum(tf.square(hypo - Y), axis=1))

##optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

##merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/Lab_12_RNN_hello_w_rnn_cell_sgkim", sess.graph)

for epoch in range(0, 1):

##    stimulus = sess.run(tf.reshape(x_data))
    stimulus = x_data
    answer = sess.run(tf.reshape(tf.one_hot(sample[1:], char_vocab_size), [-1, 4]))

    cost_val, hypo_val, rnn_cell_val = sess.run([cost, hypo, rnn_cell], feed_dict={x:stimulus, Y: answer})

    print(stimulus)
    print(answer)

    print(rnn_cell_val)
    print(hypo_val)
    print(cost_val)

##    sess.run(optimizer, feed_dict={x:stimulus, Y: answer})    

##    if step%10 == 0:
##    
##        cost_val = sess.run(cost, feed_dict={x:stimulus, Y: answer})
##
##        print(stimulus)
##        print(answer)
##
##        print(rnn_cell_val)
##        print(hypo_val)
##        print(cost_val)

##accuracy = 0.0
##
##for step in range(0, 4):
##
##    stimulus = sess.run(tf.reshape(x_data[step%4], [-1, 4]))
##    answer = sess.run(tf.reshape(tf.one_hot(sample[step%4 + 1], char_vocab_size), [-1, 4]))
##    
##    hypo_val = sess.run(hypo, feed_dict={x:stimulus})
##
##    accuracy = accuracy + tf.cast(tf.equal(hypo, answer), tf.float32)
##
##accuracy = accuracy/4
##print(accuracy)
