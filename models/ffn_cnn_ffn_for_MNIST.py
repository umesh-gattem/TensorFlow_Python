import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../tmp/data/", one_hot=True)

learning_rate = 0.01
epochs = 100
batch_size = 10
display_step = 10

no_of_inputs = 784  # no of input pixels
no_of_outputs = 10  # no of outputs
drop_out = 0.75  # keep probability

input_data = tf.placeholder(tf.float32, [None, 784], name='Input')
output_data = tf.placeholder(tf.float32, [None, 10], name='Output')
keep_prob = tf.placeholder(tf.float32)

weights = {
    'weight1': tf.Variable(tf.random_normal([784, 1024])),  # for fully connected layer
    'weight2': tf.Variable(tf.random_normal([1024, 784])),  # for second fully connected layer
    'weight3': tf.Variable(tf.random_normal([5, 5, 1, 32])),  # for 1st Conv layer C5-32
    'weight4': tf.Variable(tf.random_normal([5, 5, 32, 64])),  # for 2nd Conv layer C5-64
    'weight5': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),  # for fully connected layer
    'weight6': tf.Variable(tf.random_normal([1024, no_of_outputs]))  # for output layer
}

bias = {
    'bias1': tf.Variable(tf.random_normal([1024])),
    'bias2': tf.Variable(tf.random_normal([784])),
    'bias3': tf.Variable(tf.random_normal([32])),
    'bias4': tf.Variable(tf.random_normal([64])),
    'bias5': tf.Variable(tf.random_normal([1024])),
    'bias6': tf.Variable(tf.random_normal([no_of_outputs]))
}


def model(x, weights, bias, drop_out):
    layer1 = tf.sigmoid(tf.add(tf.matmul(x, weights['weight1']), bias['bias1']))
    layer2 = tf.sigmoid(tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2']))
    layer3 = tf.reshape(layer2, shape=[-1, 28, 28, 1])
    layer4 = tf.nn.conv2d(layer3, weights['weight3'], strides=[1, 1, 1, 1], padding='SAME')
    layer4 = tf.nn.relu(tf.nn.bias_add(layer4, bias['bias3']))
    layer5 = tf.nn.max_pool(layer4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer6 = tf.nn.conv2d(layer5, weights['weight4'], strides=[1, 1, 1, 1], padding='SAME')
    layer6 = tf.nn.relu(tf.nn.bias_add(layer6, bias['bias4']))
    layer7 = tf.nn.max_pool(layer6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer8 = tf.reshape(layer7, shape=[-1, 3136])
    layer8 = tf.nn.relu(tf.add(tf.matmul(layer8, weights['weight5']), bias['bias5']))
    layer8 = tf.nn.dropout(layer8, drop_out)
    layer9 = tf.add(tf.matmul(layer8, weights['weight6']), bias['bias6'])
    return layer9

pred = model(input_data, weights, bias, keep_prob)

# Define loss and optimizer
# cost = tf.reduce_sum(tf.pow(pred - output_data, 2)) / 2
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, output_data))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(output_data, 1))
accuracy = tf.mul(100.0, tf.reduce_mean(tf.cast(correct_pred, tf.float32)))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    for epoch in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={input_data: batch_x, output_data: batch_y,
                                       keep_prob: drop_out})
        if epoch % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={input_data: batch_x,
                                                              output_data: batch_y,
                                                              keep_prob: 1.})
            print("Iter ", epoch, " Mini batch Loss= ", "{:.3f}".format(
                loss) + ", Training Accuracy= ", "{:.3f}".format(acc))
    print("Optimization Finished!")
    print("Total test data length: ", len(mnist.test.images))
    print("Testing Accuracy :", sess.run(accuracy, feed_dict={input_data: mnist.test.images[:],
                                                              output_data: mnist.test.labels[:],
                                                              keep_prob: 1.}))
