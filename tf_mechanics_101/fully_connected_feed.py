from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

from six.moves import xrange
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import mnist

#   Basic model parameters as external flags.
FLAGS = None

def placeholder_inputs(batch_size):

    #   Args:
    #   batch_size: number of data included in a set or epoch
    #
    #   Returns:
    #   images_placeholder: Images placeholder
    #   labels_placeholder: Labels placeholder

    image_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):

    #   Args:
    #   data_set: the set of images and labels, from input_data.read_data_sets()
    #   images_pl: the images placeholder, from placeholder_inputs()
    #   labels_pl: the labels placeholder, from placeholder_inputs()
    #
    #   Returns:
    #   feed_dict: the feed dictionary mapping from placeholders to values.
    
    images_feed = data_set.next_batch(FLAGS.batch_size)
    labels_feed = data_set.next_batch(FLAGS.fake_data)

    feed_dict = {images_pl: images_feed, labels_pl: labels_feed}

    return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):

    #   Args:
    #   sess: a tensorflow session
    #   eval_correct: the tensor, which has the correct number of predictions.
    #   images_placeholder: the images placeholder
    #   labels_placeholder: the labels placeholder
    #   data_set: the set of images and labels to evaluate, from input_data.read_data_sets()

    #   Count the number of correct predictions.
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch*FLAGS.batch_size

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count = true_count + sess.run(eval_correct, feed_dict=feed_dict)

    precision = float(true_count)/num_examples
    print('Num examples: %d Num correct: %d Precision @ 1: %0.04f'
          %(num_examples, true_count, precision))

def run_training():

    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    with tf.Graph().as_default():

        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        #   Build a tensorflow computation graph
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        #   Define a cost function
        loss = mnist.loss(logits, labels_placeholder)

        #   Gradient caluculation
        train_op = mnist.training(loss, FLAGS.learning_rate)

        #   Test and Validation
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        #   Build a summary tensor
        summary = tf.summary.merge_all()

        #   Initialization
        init = tf.global_variables_initializer()

        #   Creating saver for check points during training
        saver = tf.train.Saver()

        #   Creating a tensorflow session
        sess = tf.Session()

        #   Instantiate a summary writer to output summaries and the Graph
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        #   Run the initialization
        sess.run(init)

        #   Trainig loop
        for step in xrange(FLAGS.max_steps):

            start_time = time.time()

            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

            _, loss_value = sess.run([train_op, loss], feed_dict = feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:

                print('Step %d: loss = %.2f (%.3f sec)' %(step, loss_value, duration))

                summary_str = sess.run(summary, feed_dict = feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1)%1000 = 0 or (step + 1) == FLAGS.max_steps:

                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step = step)

                print('Training Data Evaluation:')
                do_eval(sess, eval_correct, images_placeholder,
                        labels_placeholder, data_sets.train)

                print('Validation Data Evaluation:')
                do_eval(sess, eval_correct, images_placeholder,
                        labels_placeholder, data_sets.validation)

                print('Test Data Evaluation:')
                do_eval(sess, eval_correct, images_placeholder,
                        lavels_placeholder, data_sets.test)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':

    parser = argparser.ArgumentParser()
    parser.add_argument('--learning_rate',
                        type = float,
                        default = 0.01,
                        help = 'Initial learing rate')
    
    parser.add_argument('--max_steps',
                        type = int,
                        default = 2000,
                        help = 'Number of steps to run trainer.')
    
    parser.add_argument('--hidden1',
                        type = int,
                        default = 128,
                        help = 'Number of units in hidden layer 1.')
    
    parser.add_argument('--hidden2',
                        type = int,
                        default = 32,
                        help = 'Number of units in didden layer 2.')
    
    parser.add_argument('--batch_size',
                        type = int,
                        default = 100,
                        help = 'Batch size. Must divide evenly into the dataset sizes.')
    
    parser.add_argument('--input_data_dir',
                        type = str,
                        default = '/tmp/tensorflow/mnist/input_data',
                        help = 'Directory to put the input data.')
    
    parser.add_argument('--log_dir',
                        type = str,
                        default = '/tmp/tensorflow/mnist/logs/fully_connected_feed',
                        help = 'Directory to put the log data')
    
    parser.add_argument('--fake_data',
                        default = False,
                        help = 'If true, uses fake data for unit testing.',
                        action = 'store_true')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)
    
