import tensorflow as tf


class rnn_cell:

    def __init__(self, batch_size, input_num_element, output_num_element):
        self.fb = tf.Variable(tf.zeros(shape=[batch_size, output_num_element], dtype=tf.float32, name='feedback'))

        self.Wx = tf.Variable(tf.random_normal(shape=[input_num_element, output_num_element], dtype=tf.float32, name='Wx'))
        self.Wf = tf.Variable(tf.random_normal(shape=[output_num_element, output_num_element], dtype=tf.float32, name='Wf'))

##        self.Wx = tf.Variable([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0],
##                                [5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0]], dtype=tf.float32)
##        self.Wf = tf.Variable([[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0],
##                               [6.0, 6.0, 6.0, 6.0], [8.0, 8.0, 8.0, 8.0]], dtype=tf.float32)
        
        self.stimulus = tf.placeholder(tf.float32, shape=[batch_size, input_num_element])
        self.answer = tf.placeholder(tf.float32, shape=[batch_size, output_num_element])

##        fb_loop = tf.assign(self.fb, tf.tanh(tf.matmul(self.fb, self.Wf) + tf.matmul(self.stimulus, self.Wx)), name='feedback_loop')
        self.fb_loop = tf.assign(self.fb, tf.matmul(self.fb, self.Wf) + tf.matmul(self.stimulus, self.Wx), name='feedback_loop')
##        fb_loop = tf.assign(self.fb, tf.matmul(self.fb, self.Wf) + tf.matmul(self.stimulus, self.Wx), name='feedback_loop')
        
        self.Wy = tf.Variable(tf.random_normal(shape=[output_num_element, output_num_element], dtype=tf.float32, name='Wy'))
##        self.Wy = tf.Variable([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0],
##                               [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]], dtype=tf.float32)

##        self.hypo = tf.matmul(self.fb_loop, self.Wy)
##        self.hypo = tf.nn.softmax(tf.matmul(self.fb_loop, self.Wy))
        self.hypo = tf.nn.tanh(tf.matmul(self.fb_loop, self.Wy))

        self.cost = tf.reduce_mean(tf.reduce_mean(tf.square(self.hypo - self.answer), axis=1))
##        self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.answer, logits=self.fb)

        self.learning_rate = tf.placeholder(tf.float32)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        
    def run(self, tf_sess, stimulus):

        return tf_sess.run(self.hypo, feed_dict={self.stimulus: stimulus})


    def optimize(self, tf_sess, learning_rate, stimulus, answer):
        
        tf_sess.run(self.optimizer, feed_dict={self.stimulus: stimulus, self.answer: answer, self.learning_rate: learning_rate})


    def get_cost(self, tf_sess, stimulus, answer):

        return tf_sess.run(self.cost, feed_dict={self.stimulus: stimulus, self.answer: answer})


##    def get_cost_summary(self):
##
##        return tf.summary("cost", self.cost)
