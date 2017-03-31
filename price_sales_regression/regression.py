
import tensorflow as tf
import numpy as np

def prediction(X):

    W = tf.Variable(np.random.randn(), name='weight')
    b = tf.Variable(np.random.randn(), name='bias')

    Y = tf.add(tf.multiply(X, W), b)

    return Y

def cost(prediction, sales_data):

    cost = tf.sqrt(tf.reduce_mean(tf.pow(prediction - sales_data, 2)))
    
    return cost

def training(cost, learning_rate):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    return optimizer

