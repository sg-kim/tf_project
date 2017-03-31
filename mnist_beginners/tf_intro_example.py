import numpy as np

tensor1d_x = np.array([1.0, 2.0, 3.0, 4.0])
tensor1d_y = np.array([1.0, 2.0, 3.0, 4.0])

a = tensor1d_x*tensor1d_y

import tensorflow as tf

#b = tf.reduce_sum(tensor1d_x*tensor1d_y, reduction_indices = [0])
b = tf.reduce_sum(tensor1d_x*tensor1d_y, 0)

sess = tf.Session()

c = sess.run(b)

print(tensor1d_x.shape)
print(tensor1d_y.shape)

print(a)
print(b)
print(c)

