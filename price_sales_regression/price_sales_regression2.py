import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def normalize(array):
    ret_val = (array - array.mean())/array.std()
    return ret_val

######################################################################
##      Input data
######################################################################

sales_data = np.array([2142, 1405, 3423, 3323, 1182, 840, 2231, 1100,
                       1329, 4329, 2293, 1120, 4120, 3320, 2231, 759,
                       2231, 1982, 989, 1359, 2521, 457, 898, 1179,
                       538, 3412, 2234, 522, 1775, 2789, 1446, 1623,
                       2231, 4672, 3212, 1156, 1721, 2981, 1992, 1020,
                       1420, 2241, 4521, 3240, 897, 1123, 962, 1729])

price_data = np.array([342, 673, 302, 241, 520, 550, 420, 428,
                       289, 211, 456, 389, 321, 352, 411, 498,
                       521, 301, 241, 205, 409, 524, 508, 450,
                       789, 300, 388, 802, 556, 478, 490, 552,
                       443, 120, 229, 1120, 530, 400, 421, 458,
                       430, 320, 248, 423, 239, 506, 478, 330])

######################################################################
##      Regression model
######################################################################

#X = tf.placeholder(tf.float32, [1, None])
#Y_ = tf.placeholder(tf.float32, [1, None])
X = tf.placeholder(tf.float32)
Y_ = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

Y = tf.add(tf.multiply(X, W), b)

######################################################################
##      Cost function and Optimizer definition
######################################################################

#cost = tf.sqrt(tf.reduce_sum(tf.pow((Y - Y_), 2)) / sales_data.size)
cost = tf.sqrt(tf.reduce_mean(tf.pow((Y - Y_), 2)))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

######################################################################
##      Initialization
######################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sales_data_n = sess.run(tf.cast(normalize(sales_data), tf.float32))
price_data_n = sess.run(tf.cast(normalize(price_data), tf.float32))

######################################################################
##      Training
######################################################################

for step in range(200):

    for i, j in zip(price_data_n, sales_data_n):
    #for i, j in zip(price_data, sales_data):
        sess.run(optimizer, feed_dict = {X: i, Y_: j})

    #sess.run(optimizer, feed_dict = {X: price_data_n, Y_: sales_data_n})

    if step % 10 == 0:
        print("Iteration: %d cost: %f"
              %(step, sess.run(cost, feed_dict = {X: price_data_n, Y_: sales_data_n})))
        #print("Iteration: %d cost: %f"
        #      %(step, sess.run(cost, feed_dict = {X: price_data, Y_: sales_data})))

######################################################################
##      Evaluation
######################################################################

print("Final cost: %f, W: %f, b: %f (Y = W * X + b)"
      %(sess.run(cost, feed_dict = {X: price_data_n, Y_: sales_data_n}),
        sess.run(W),
        sess.run(b)))

#print("Final cost: %f, W: %f, b: %f (Y = W * X + b)"
#      %(sess.run(cost, feed_dict = {X: price_data, Y_: sales_data}),
#        sess.run(W),
#        sess.run(b)))


#plt.plot(sales_data_n, color='blue', marker='o', linestyle='none', label ='Normalized sales samples')
#plt.plot(price_data_n, color='red', marker='o', linestyle='none', label ='Normalized price samples')
#plt.plot(sales_data_n, price_data_n, color='green', marker='o', linestyle='none', label ='Price-Sales')
plt.plot(price_data_n, sales_data_n, color='green', marker='o', linestyle='none', label ='Price-Sales')
plt.plot(price_data_n, sess.run(W)*price_data_n + sess.run(b), label = 'Fitted line')
#plt.plot(sess.run(W)*price_data_n + sess.run(b), price_data_n, label = 'Fitted line')
#plt.plot(sales_data, price_data, color='green', marker='o', linestyle='none', label ='Price-Sales')
#plt.plot(price_data, sess.run(W)*price_data + sess.run(b), label = 'Fitted line')
plt.legend()
plt.show()

