import tensorflow as tf

weight1 = tf.Variable(tf.random_normal([16, 4]))
weight2 = tf.Variable(tf.random_normal([6, 4]))
weight3, weight4, weight5 = tf.split(0, 3, weight2)
# print(weight1)
# print(weight2)
# print(weight3)
# print(weight4)
# print(weight5)
weight6 = tf.concat(0, [weight1, weight2])
print(weight6)