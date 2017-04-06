import tensorflow as tf


##def rnn_cell(batch_size, input_num_element, output_num_element, stimulus):
##    """Set the parameter initialization using the method described.
##    This method is designed to generate a RNN cell with given properties.
##    Seunggu Kim(Apr. 3, 2017):
##    	For better understanding of RNN. Enjoy:-)
##    Args:
##    	batch_size: number of samples in a batch.
##    	input_num_element: number of elements in a sample.
##    	output_num_element: number of elements in an output.
##    	stimuls: sample input. for example, placeholder x.
##    Returns:
##	An RNN cell.
##    """
##    h = tf.Variable(tf.zeros(shape=[batch_size, input_num_element], dtype=tf.float32))
##
##    Wxh = tf.Variable(tf.random_normal(shape=[input_num_element, output_num_element], dtype=tf.float32))
##    Whh = tf.Variable(tf.random_normal(shape=[input_num_element, output_num_element], dtype=tf.float32))
####    Wxh = tf.Variable([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0],
####                   [5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0]], name='Weight_xh')
####    Whh = tf.Variable([[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0],
####                   [6.0, 6.0, 6.0, 6.0], [8.0, 8.0, 8.0, 8.0]], name='Weight_hh')
##
##    stimulus = tf.cast(stimulus, dtype=tf.float32)
##
##    h = h.assign(tf.tanh(tf.matmul(h, Whh) + tf.matmul(stimulus, Wxh)))
##
##    Wyh = tf.Variable(tf.random_normal(shape=[output_num_element, output_num_element], dtype=tf.float32))
####    Wyh = tf.Variable([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0],
####                   [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]], name='Weight_yh')
##
##    return tf.matmul(h, Wyh)


class rnn_cell:

    def __init__(self, batch_size, input_num_element, output_num_element):
        self.h = tf.Variable(tf.zeros(shape=[batch_size, output_num_element], dtype=tf.float32, name='h'))

        self.Wxh = tf.Variable(tf.random_normal(shape=[input_num_element, output_num_element], dtype=tf.float32, name='Wxh'))
        self.Whh = tf.Variable(tf.random_normal(shape=[output_num_element, output_num_element], dtype=tf.float32, name='Whh'))

##        self.Wxh = tf.Variable([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0],
##                                [5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0]], dtype=tf.float32)
##        self.Whh = tf.Variable([[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0],
##                               [6.0, 6.0, 6.0, 6.0], [8.0, 8.0, 8.0, 8.0]], dtype=tf.float32)
        
        self.stimulus = tf.placeholder(tf.float32, shape=[batch_size, input_num_element])
        self.answer = tf.placeholder(tf.float32, shape=[batch_size, output_num_element])

        self.h = self.h.assign(tf.tanh(tf.matmul(self.h, self.Whh) + tf.matmul(self.stimulus, self.Wxh)))
##        self.h = self.h.assign(tf.matmul(self.h, self.Whh) + tf.matmul(self.stimulus, self.Wxh))

        self.Wyh = tf.Variable(tf.random_normal(shape=[output_num_element, output_num_element], dtype=tf.float32, name='Wyh'))
##        self.Wyh = tf.Variable([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0],
##                                [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]], dtype=tf.float32)

        self.hypo = tf.matmul(self.h, self.Wyh)

        self.cost = tf.reduce_mean(tf.reduce_mean(tf.square(self.hypo - self.answer), axis=1))

        self.learning_rate = tf.placeholder(tf.float32)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        
    def run(self, tf_sess, stimulus):

        return tf_sess.run(self.hypo, feed_dict={self.stimulus: stimulus})


    def optimize(self, tf_sess, learning_rate, stimulus, answer):
        
        tf_sess.run(self.optimizer, feed_dict={self.stimulus: stimulus, self.answer: answer, self.learning_rate: learning_rate})

