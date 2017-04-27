import tensorflow as tf

##################################################################
##  Update variables
##################################################################

acc_val = tf.Variable(0)
inc = tf.constant(1)

new_val = tf.add(acc_val, inc)
update = tf.assign(acc_val, new_val)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(0, 3):
    run_val = sess.run(update)
    print("step %d: %d" %(step, run_val))

##################################################################
##  name_scope
##################################################################

with tf.name_scope("name_scope_1"):
    init = tf.constant_initializer(value = 1)
    var1 = tf.get_variable(name="var1", shape = [1], initializer = init)
    var2 = tf.Variable(name="var2", initial_value = 2)
    var3 = tf.Variable(name="var3", initial_value = 3)
    var4 = tf.Variable(name="var4", initial_value = 4)

    tf.summary.histogram("var1", var1)
    tf.summary.scalar("var2", var2)
    tf.summary.scalar("var3", var3)
    tf.summary.scalar("var4", var4)

sess.run(tf.global_variables_initializer())

print(var1.name)
print(sess.run(var1))
print(var2.name)
print(sess.run(var2))
print(var3.name)
print(sess.run(var3))
print(var4.name)
print(sess.run(var4))

with tf.name_scope("name_scope_1") as scope:
##    scope.reuse()
##    v1 = tf.get_variable(name="var1", shape = [1])
    var2_reuse = tf.get_variable(name="var2", shape = [1])
##    x1 = tf.add(v1, v2)
    var3_reuse = tf.get_variable(name="var3", shape = [1])  
    x2 = tf.add(var2_reuse, var3_reuse)

    tf.summary.histogram("var2_reuse", var2_reuse)  
    tf.summary.histogram("var3_reuse", var3_reuse)
    tf.summary.histogram("x2", x2)

sess.run(tf.global_variables_initializer())

##print(v1.name)
##print(sess.run(v1))
print(var2_reuse.name)
print(sess.run(var2_reuse))
##print(x1.name)
##print(sess.run(x1))
print(var3_reuse.name)
print(sess.run(var3_reuse))
print(x2.name)
print(sess.run(x2))

with tf.variable_scope("var_scope_1"):
    init = tf.constant_initializer(value = 5)
    var5 = tf.get_variable(name="var5", shape = [1], initializer = init)
##    var6 = tf.Variable(name="var6", initial_value = 6)
    init = tf.constant_initializer(value = 6)
    var6 = tf.get_variable(name="var6", shape = [1], initializer = init)
    var7 = tf.Variable(name="var7", initial_value = 7)
    var8 = tf.Variable(name="var8", initial_value = 8)

    tf.summary.histogram("var5", var5)
    tf.summary.histogram("var6", var6)
    tf.summary.scalar("var7", var7)
    tf.summary.scalar("var8", var8)


sess.run(tf.global_variables_initializer())

print(var5.name)
print(sess.run(var5))
print(var6.name)
print(sess.run(var6))
print(var7.name)
print(sess.run(var7))
print(var8.name)
print(sess.run(var8))

with tf.variable_scope("var_scope_1") as scope:
    scope.reuse_variables()
##    v1 = tf.get_variable(name="var1", shape = [1])
    var5_reuse = tf.get_variable(name="var5", shape = [1])
##    x1 = tf.add(v1, v2)
    var6_reuse = tf.get_variable(name="var6", shape = [1])
    x5 = tf.add(var5_reuse, var6_reuse)

    tf.summary.histogram("var5_reuse", var5_reuse)
    tf.summary.histogram("var6_reuse", var6_reuse)
    tf.summary.histogram("x5", x5)

sess.run(tf.global_variables_initializer())

##print(v1.name)
##print(sess.run(v1))
print(var5_reuse.name)
print(sess.run(var5_reuse))
##print(x1.name)
##print(sess.run(x1))
print(var6_reuse.name)
print(sess.run(var6_reuse))
print(x5.name)
print(sess.run(x5))


##init = tf.constant_initializer(value = 9)
##var9 = tf.get_variable(name="var9", shape = [1], initializer = init)
##
##var9_reuse = tf.get_variable(name="var9", shape = [1])
##
##sess.run(tf.global_variables_initializer())
##
##print(var9.name)
##print(sess.run(var9))
##print(var9_reuse.name)
##print(sess.run(var9_reuse))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/tf_var_handling_basics", sess.graph)

summary = sess.run(merged)
writer.add_summary(summary)
