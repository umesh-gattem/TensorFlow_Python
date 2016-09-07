import tensorflow as tf

a = tf.constant(45)
b = tf.constant(78)
addition = tf.add(a, b)

with tf.Session() as sess:
    output = sess.run(addition)
    print("Addition is : ", output)

print("Addition is : ", output)

