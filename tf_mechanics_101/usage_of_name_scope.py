import tensorflow as tf

def my_name_scopes(scope_name):

    with tf.name_scope('hidden1'):
        weight = tf.constant(1.0, tf.float32)
        var = tf.Variable(1.1, dtype=tf.float32)

    with tf.name_scope("hidden2"):
        weight = tf.constant(2.0, tf.float32)
        var = tf.Variable(2.2, dtype=tf.float32)

    with tf.name_scope("hidden3"):
        weight = tf.constant(3.0, tf.float32)
        var = tf.Variable(3.3, dtype=tf.float32)

    if scope_name == 'hidden1':
        ret_val = {weight: hidden1/weight, var: hidden1/var}
    elif scope_name == 'hidden2':
        ret_val = {weight: hidden2/weight, var: hidden2/var}
    elif scope_name == 'hidden3':
        ret_val = {weight: hidden3/weight, var: hidden3/var}
    else:
        ret_val = 'none'

    return ret_val

items = my_name_scopes('hidden1')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

weight_var = sess.run(items)

print(weight_var)

#print(hidden3/weight)


#my_name_scopes()

