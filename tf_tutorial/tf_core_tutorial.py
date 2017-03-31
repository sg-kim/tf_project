import numpy as np
import tensorflow as tf
import time

#######################################################################
##          Model parameters
#######################################################################

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

#######################################################################
##          Model input and output
#######################################################################

x = tf.placeholder(tf.float32)
linear_model = W*x + b

y = tf.placeholder(tf.float32)

#######################################################################
##          Cost
#######################################################################

cost = tf.reduce_sum(tf.square(linear_model - y))   #   Sum of squares

#######################################################################
##          Optimizer
#######################################################################

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

#######################################################################
##          Training data
#######################################################################

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

#######################################################################
##          Training
#######################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = time.time()

for i in range(1000):
    if(i%100 == 0):
        print("Train step %d cost = %g"%(i, sess.run(cost, feed_dict={x:x_train, y:y_train})))
    sess.run(train, {x:x_train, y:y_train})
    
#######################################################################
##          Evaluation
#######################################################################

print("Cost: %f"%sess.run(cost, feed_dict={x:x_train, y:y_train}))

end_time = time.time()

print("Run time: %f"%(end_time - start_time))

