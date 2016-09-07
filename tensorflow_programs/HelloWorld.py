import tensorflow as tf

string = tf.constant("Hello World")
sess = tf.Session()
print(sess.run(string))
sess.close()