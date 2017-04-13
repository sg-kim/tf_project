import tensorflow as tf
import numpy as np

sales_data = np.array([[2142, 1405, 3423, 3323, 1182, 840, 2231, 1100,
                       1329, 4329, 2293, 1120, 4120, 3320, 2231, 759,
                       2231, 1982, 989, 1359, 2521, 457, 898, 1179,
                       538, 3412, 2234, 522, 1775, 2789, 1446, 1623,
                       2231, 4672, 3212, 1156, 1721, 2981, 1992, 1020,
                       1420, 2241, 4521, 3240, 897, 1123, 962, 1729]])

price_data = np.array([[342, 673, 302, 241, 520, 550, 420, 428,
                       289, 211, 456, 389, 321, 352, 411, 498,
                       521, 301, 241, 205, 409, 524, 508, 450,
                       789, 300, 388, 802, 556, 478, 490, 552,
                       443, 120, 229, 1120, 530, 400, 421, 458,
                       430, 320, 248, 423, 239, 506, 478, 330]])

def normalize(array):
    ret_val = (array - array.mean())/array.std()
    return ret_val

price_data_n = normalize(price_data).astype(np.float32)
sales_data_n = normalize(sales_data).astype(np.float32)

price = tf.placeholder(tf.float32, [1, None])

y = 2*price

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#price_data_n2 = sess.run(tf.reshape(price_data_n, [1, -1]))

print(sess.run(y, feed_dict={price: price_data_n}))
