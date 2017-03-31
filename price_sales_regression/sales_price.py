
import argparse
import sys
import time

import tensorflow as tf
import numpy as np

import regression

FLAGS = None

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

def normalize(array):
    ret_val = (array - array.mean())/array.std()
    return ret_val

def placeholder_inputs():

    price_placeholder = tf.placeholder(tf.float32)
    sales_placeholder = tf.placeholder(tf.float32)

    return price_placeholder, sales_placeholder

def make_feed_dict(price, sales, price_data, sales_data):

    price_data_n = normalize(price_data).astype(np.float32)
    sales_data_n = normalize(sales_data).astype(np.float32)

    feed_dict = {price: price_data_n, sales: sales_data_n}

    return feed_dict

def do_eval(sess, cost, prediction, sales_data):

    print("cost: %f"%(sess.run(cost(prediction, sales_data))))

def do_plot(sess, prediction, price_data, sales_data):

    plt.plot(price_data, sales_data, color='green', marker='o', linestyle='none', label ='Price-Sales')
    plt.plot(price_data, sess.run(prediction, feed_dict = {price_data: price_data}), label = 'Fitted line')
    plt.legend()
    plt.show()

def run_training():

    with tf.Graph().as_default():

        price_placeholder, sales_placeholder = placeholder_inputs()

        prediction = regression.prediction(price_placeholder)

        cost = regression.cost(prediction, sales_placeholder)

        train = regression.training(cost, FLAGS.learning_rate)

        init = tf.global_variables_initializer()

        sess = tf.Session()

        sess.run(init)

        for step in range(FLAGS.max_steps):

            start_time = time.time()

            feed_dict = make_feed_dict(price_placeholder, sales_placeholder, price_data, sales_data)

            for i, j in zip(feed_dict[price_placeholder], feed_dict[sales_placeholder]):
                sess.run(train, feed_dict = {price_placeholder: i, sales_data: j})

            if step%10 == 0:

                do_eval(sess, cost, prediction, feed_dict[price_placeholder], feed_dict[sales_placeholder])

        end_time = time.time()

        do_plot(sess, prediction, feed_dict[price_placeholder], feed_dict[sales_placeholder])

        print('Running time: %f'%(start_time - end_time))

def main(argv):        
    run_training()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate for optimizer')

    parser.add_argument('--max_steps', type = int, default = 10, help = 'Number of training steps')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main = main, argv=[sys.argv[0]] + unparsed)
    
