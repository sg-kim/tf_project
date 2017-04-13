import random
import tensorflow as tf

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

file = []
file.append('.\cifar-10-batches-py\data_batch_1')
file.append('.\cifar-10-batches-py\data_batch_2')
file.append('.\cifar-10-batches-py\data_batch_3')
file.append('.\cifar-10-batches-py\data_batch_4')
file.append('.\cifar-10-batches-py\data_batch_5')



dict_data = unpickle(file[0])

data_list, label_list = dict_batch(dict_data, 2)


x = tf.placeholder(tf.float32, shape=[None, 3072])
Y = tf.placeholder(tf.float32, shape=[None, 10])

w_conv1 = tf.Variable(tf.random_normal([5, 5, 3, 45]))
b_conv1 = tf.Variable(tf.random_normal([1024]))

x_image = tf.reshape(x, [-1, 32, 32, 1])

h_conv1 = tf.nn.conv2d(x_image, w_conv1, stride=[1, 1, 1, 1], use_cudnn_on_gpu = cudnn_on_gpu, padding='SAME')
