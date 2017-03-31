from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

#   The MNIST dataset has 10 classes, which represent the digits 0 through 9.
NUM_CLASSES = 10

#   The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units):

    #   Arguments:
    #   images: image placeholder, from inputs().
    #   hidden1_units: size of the first hidden layer.
    #   hidden2_units: size of the second hidden layer.
    #
    #   Returns:
    #   softmax_linear: output tensor with the computed logits.

    ###############################################################
    #   The first hidden layer
    ###############################################################
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS, hidden1_units],
                stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),
            name='weights')

        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')

        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    ###############################################################
    #   The second hidden layer
    ###############################################################
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden1_units, hidden2_units],
                stddev=1.0/math.sqrt(float(hidden1_units))),
            name='weights')

        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')

        logits = tf.matmul(hidden1, weights) + biases

    ###############################################################
    #   Applying linear activation function
    ###############################################################
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal(
                [hidden2_units, NUM_CLASSES],
                stddev=1.0/math.sqrt(float(hidden2_units))),
            name='weights')

        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')

        logits = tf.matmul(hidden2, weights) + biases

    return logits

def loss(logits, labels):

    #   Args:
    #   logits: logits tensor, float, [batch_size, NUM_CLASSES]
    #   labels: digit labels tensor, int, [batch_size]
    #
    #   Returns:
    #   loss: loss tensor of type float

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='xentropy')

    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):

    #   Args:
    #   loss: loss tensor, from loss()
    #   learning_rate: the learning rate to use for gradient descent.
    #
    #   Returns:
    #   train_op: the Op for training

    #   Outputs a Summary protocol buffer containing a single scalar value
    #   Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)

    #   Create the gradient descent optimizer with given learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    #   Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    #   Use the optimizer to apply the gradient that minimize the loss
    #   and also increment the global step counter as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):

    #   Args:
    #   logits: logits tensor, float, [batch_size, NUM_CLASSES]
    #   labels: labels tensor, int32, [batch_size], with values in the range [0, NUM_CLASSES).
    #
    #   Returns:
    #   A scalar int32 tensor with the number of examples out of batch_size that were predicted correctly.

    #   in_top_k Op returns a bool tensor with shape [batch_size] that is true for the examples
    #   where the label is in the top k (here k=1) of all logits for that example.

    correct = tf.nn.in_top_k(logits, labels, 1)

    #   Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
