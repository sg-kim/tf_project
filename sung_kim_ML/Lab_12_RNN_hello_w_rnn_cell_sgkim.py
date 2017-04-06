import tensorflow as tf
import numpy as np

import rnn_cell_sgkim as rcell

import time

char_rdic = ['h', 'e', 'l', 'o']        #   raw dictionary, idex - value
char_dic = {w: i for i, w in enumerate(char_rdic)}      #   value - index
x_data = np.array([[1, 0, 0, 0],                    #   h
                   [0, 1, 0, 0],                    #   e
                   [0, 0, 1, 0],                    #   l
                   [0, 0, 1, 0]], dtype=float)      #   l

sample = [char_dic[c] for c in 'hello']

x = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 4])

char_vocab_size = len(char_dic)
time_step_size = char_vocab_size

rnn_cell = rcell.rnn_cell(1, 4, 4)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

##cost_summ = rnn_cell.get_cost_summary()
##
##merged = tf.summary.merge_all()
##writer = tf.summary.FileWriter("./logs/Lab_12_RNN_hello_w_rnn_cell_sgkim", sess.graph)

start_time = time.time()

for epoch in range(0, 101):

    for step in range(0, char_vocab_size):

        stimulus = sess.run(tf.reshape(x_data[step], [-1, 4]))

        answer = sess.run(tf.reshape(tf.one_hot(sample[step + 1], char_vocab_size), [-1, 4]))

        rnn_cell.optimize(sess, 0.0005, stimulus, answer)

##    summary = sess.run(merged)
##    writer.add_summary(summary, epoch)
    
    cost_val = rnn_cell.get_cost(sess, stimulus, answer)

    print("epoch: %d, cost: %f"%(epoch, cost_val))

end_training = time.time()

accuracy = 0.0

for step in range(0, 4):

    stimulus = sess.run(tf.reshape(x_data[step], [-1, 4]))
    answer = sess.run(tf.reshape(tf.one_hot(sample[step + 1], char_vocab_size), [-1, 4]))

    hypo_val = rnn_cell.run(sess, stimulus)
    hypo_val_one_hot = sess.run(tf.one_hot(tf.arg_max(hypo_val, 1), 4))

    print(hypo_val)
    print(hypo_val_one_hot)
    print(answer)

    accuracy = accuracy +sess.run(
        tf.cast(
            tf.equal(
                tf.arg_max(hypo_val_one_hot, 1), tf.arg_max(answer, 1)),
            tf.float32))

accuracy = accuracy/4
print("accuracy: %f"%(accuracy))

end_run = time.time()

print("Training time: %f"%(end_training - start_time))
print("Evaluation time: %f"%(end_run - end_training))
