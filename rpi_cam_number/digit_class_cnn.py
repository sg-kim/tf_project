import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class digit_classifier:

    cudnn_on_gpu = False

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]), name="w_conv1")
    b_conv1 = tf.Variable(tf.random_normal([32]), name="b_conv1")

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], use_cudnn_on_gpu=cudnn_on_gpu, padding='SAME')
    h1 = tf.nn.relu(h_conv1 + b_conv1)
    h_pool1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]), name="w_conv2")
    b_conv2 = tf.Variable(tf.random_normal([64]), name="b_conv1")

    h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], use_cudnn_on_gpu=cudnn_on_gpu, padding='SAME')
    h2 = tf.nn.relu(h_conv2 + b_conv2)
    h_pool2 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_fc1 = tf.Variable(tf.random_normal([7*7*64, 1024]), name="w_fc1")
    b_fc1 = tf.Variable(tf.random_normal([1024]), name="b_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.random_normal([1024, 10]), name="w_fc2")
    b_fc2 = tf.Variable(tf.random_normal([10]), name="b_fc2")

    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y = tf.nn.softmax(h_fc2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_fc2))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def __init__(self, cudnn_on_gpu = False):

        self.cudnn_on_gpu = cudnn_on_gpu

    def run(self, input_img, tf_sess):

        input_img_flat = tf_sess.run(tf.reshape(input_img, [-1, 784]))
        prediction = tf_sess.run(self.y, feed_dict={self.x: input_img_flat, self.keep_prob: 1.0})

        return prediction

    def train(self, tf_sess, n_epoch):

        import time

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        start_training = time.time()

        for step in range(0, n_epoch):

            batch = mnist.train.next_batch(50)
            tf_sess.run(self.optimizer, feed_dict={self.x: batch[0], self.y_:batch[1], self.keep_prob: 0.5})

            if step%20 == 0:
                train_accuracy = tf_sess.run(self.accuracy, feed_dict={self.x: batch[0], self.y_:batch[1], self.keep_prob: 1.0})
                print("train step %d, accuracy %g" %(step, train_accuracy))

        end_training = time.time()

        print("Training finished. time for training: %d sec" %(end_training - start_training))

    def save_weights(self, tf_sess, path):

        saver = tf.train.Saver({"w_conv1":self.W_conv1, "b_conv1":self.b_conv1,
                                "w_conv2":self.W_conv2, "b_conv2":self.b_conv2,
                                "w_fc1":self.W_fc1, "b_fc1":self.b_fc1,
                                "w_fc2":self.W_fc2, "b_fc2":self.b_fc2})

        save_path = saver.save(tf_sess, path)

        return save_path

    def restore_weights(self, tf_sess, path):

        saver = tf.train.Saver({"w_conv1":self.W_conv1, "b_conv1":self.b_conv1,
                                "w_conv2":self.W_conv2, "b_conv2":self.b_conv2,
                                "w_fc1":self.W_fc1, "b_fc1":self.b_fc1,
                                "w_fc2":self.W_fc2, "b_fc2":self.b_fc2})

        saver.restore(tf_sess, path)

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        batch = mnist.train.next_batch(50)
        predict_accuracy = tf_sess.run(self.accuracy, feed_dict={self.x: batch[0], self.y_:batch[1], self.keep_prob: 1.0})
        print("weights restored. accuracy %g" %(predict_accuracy))
