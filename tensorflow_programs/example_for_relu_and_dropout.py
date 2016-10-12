import tensorflow as tf

weight = tf.Variable(tf.random_normal([2, 4]), name='weight')
relu_weight = tf.nn.relu(weight)
softmax_weight = tf.nn.softmax(relu_weight)
softmax_weight_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(softmax_weight, softmax_weight)
change_in_weight = tf.nn.dropout(relu_weight, .5)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(weight))
    # print(sess.run(relu_weight))
    print(sess.run(softmax_weight))
    print(sess.run(softmax_weight_cross_entropy))
    # print(sess.run(change_in_weight))
