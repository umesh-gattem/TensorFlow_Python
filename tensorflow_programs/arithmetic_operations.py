import tensorflow as tf

a = tf.constant(10)
c = tf.constant(45)

sess = tf.Session()
multiply = tf.mul(a, c)
addition = tf.add(a, c)
subtract = tf.sub(a, c)
division = tf.div(c, a)
print("Multiplication is : ", sess.run(multiply))
print("Adding is : ", sess.run(addition))
print("Subtraction is : ", sess.run(subtract))
print("Division is : ", sess.run(division))
sess.close()
