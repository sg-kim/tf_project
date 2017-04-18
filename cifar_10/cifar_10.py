import random
import tensorflow as tf
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def dict_batch(dict_data, batch_size):
    dict_len = len(dict_data[b'labels'])

    if(batch_size < dict_len):
        begin = random.randrange(0, dict_len - batch_size)
    
        data_list = []
        label_list = []

        for idx in range(begin, begin + batch_size):
            data_list.append(dict_data[b'data'][idx])
            label_list.append(dict_data[b'labels'][idx])

        return [data_list, label_list]

    else:
        return [-1, -1]

cudnn_on_gpu = True


x = tf.placeholder(tf.float32, shape=[None, 3072])
Y = tf.placeholder(tf.uint8, shape=[None, 1])

W_conv1 = tf.Variable(tf.random_normal([5, 5, 3, 15]))
b_conv1 = tf.Variable(tf.random_normal([15]))

x_image = tf.reshape(x, [-1, 32, 32, 3])

h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], use_cudnn_on_gpu = cudnn_on_gpu, padding='SAME')
h_1 = tf.nn.relu(h_conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_conv2 = tf.Variable(tf.random_normal([5, 5, 15, 45]))
b_conv2 = tf.Variable(tf.random_normal([45]))

h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], use_cudnn_on_gpu = cudnn_on_gpu, padding='SAME')
h_2 = tf.nn.relu(h_conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_fc1 = tf.Variable(tf.random_normal([8*8*45, 512]))
b_fc1 = tf.Variable(tf.random_normal([512]))

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*45])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob_fc1 = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_fc1)


W_fc2 = tf.Variable(tf.random_normal([512, 128]))
b_fc2 = tf.Variable(tf.random_normal([128]))

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

keep_prob_fc2 = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_fc2)


W_fc3 = tf.Variable(tf.random_normal([128, 10]))
b_fc3 = tf.Variable(tf.random_normal([10]))

h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
hypo = tf.nn.softmax(h_fc3)

prediction = tf.cast(tf.arg_max(hypo, 1), tf.uint8)
correct_prediction = tf.cast(tf.equal(prediction, Y), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

one_hot_Y = tf.cast(tf.one_hot(Y, 10), tf.float32)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_Y, logits=h_fc3))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)


file = []
file.append('.\\cifar-10-batches-py\\data_batch_1')
file.append('.\\cifar-10-batches-py\\data_batch_2')
file.append('.\\cifar-10-batches-py\\data_batch_3')
file.append('.\\cifar-10-batches-py\\data_batch_4')
file.append('.\\cifar-10-batches-py\\data_batch_5')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

file_sel = random.randrange(0, 4)
dict_data = unpickle(file[file_sel])

data_list, label_list = dict_batch(dict_data, 2)
data_list = sess.run(tf.reshape(data_list, [3072, -1]))
##label_list = sess.run(tf.reshape(label_list, [-1, 1]))


##for step in range(0, 1000):
##
##    file_sel = random.randrange(0, 4)
##    dict_data = unpickle(file[file_sel])
##
##    data_list, label_list = dict_batch(dict_data, 50)
##    label_list = sess.run(tf.reshape(label_list, [-1, 1]))
##
##    sess.run(optimizer, feed_dict={x:data_list, Y:label_list, keep_prob_fc1:0.5, keep_prob_fc2:0.5})
##
##    if step%20 == 0:
##        cost_val, acc_val = sess.run([cross_entropy, accuracy], feed_dict={x:data_list, Y:label_list, keep_prob_fc1:1.0, keep_prob_fc2:1.0})
##        print('step: %d cost: %.3f accuracy: %.3f'%(step, cost_val, acc_val))
##
##test_file = '.\\cifar-10-batches-py\\test_batch'
##test_data = unpickle(test_file)
##test_list, test_label = dict_batch(test_data, 500)
##test_label = sess.run(tf.reshape(test_label, [-1, 1]))
##
##acc_val = sess.run(accuracy, feed_dict={x:test_list, Y:test_label, keep_prob_fc1:1.0, keep_prob_fc2:1.0})
##print('Test result: %f'%(acc_val))
