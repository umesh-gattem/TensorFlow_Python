import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
epochs = 100
batch_size = 10
display_step = 1
logs_path = 'cnn_for_mnist_data'

no_of_inputs = 784  # no of input pixelst
no_of_outputs = 10  # no of outputs
drop_out = 0.75  # keep probability

input_data = tf.placeholder(tf.float32, [None, 784], name='Input')
output_data = tf.placeholder(tf.float32, [None, 10], name='Output')
keep_prob = tf.placeholder(tf.float32)

weights = {
    'weight1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='weight1'),  # for 1st Conv layer C5-32
    'weight2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='weight2'),  # for 2nd Conv layer C5-64
    'weight3': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name='weight3'),  # for fully connected layer
    'weight4': tf.Variable(tf.random_normal([1024, no_of_outputs]), name='weight4')  # for output layer
}

bias = {
    'bias1': tf.Variable(tf.random_normal([32]), name='bias1'),
    'bias2': tf.Variable(tf.random_normal([64]), name='bias2'),
    'bias3': tf.Variable(tf.random_normal([1024]), name='bias3'),
    'bias4': tf.Variable(tf.random_normal([no_of_outputs]), name='bias4')
}


def conv(x, W, b, strides=1):
    # Conv wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def find_maxpool(x, k=2):
    # MaxPool wrapper
    max_pool = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return max_pool


def conv_layer(x, weights, bias, drop_out):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # First Convolution Layer
    conv1 = conv(x, weights['weight1'], bias['bias1'])
    # Max Pooling
    conv1 = find_maxpool(conv1, k=2)

    # Second Convolution Layer
    conv2 = conv(conv1, weights['weight2'], bias['bias2'])
    # Max Pooling
    conv2 = find_maxpool(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['weight3'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['weight3']), bias['bias3'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, drop_out)

    # Output, class prediction
    output = tf.add(tf.matmul(fc1, weights['weight4']), bias['bias4'])
    return output


with tf.name_scope("Prediction"):
    pred = conv_layer(input_data, weights, bias, keep_prob)

# Define loss and optimizer
# cost = tf.reduce_sum(tf.pow(pred - output_data, 2)) / 2
with tf.name_scope("Loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, output_data))

with tf.name_scope("Optimization"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope("Accuracy"):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(output_data, 1))
    accuracy = tf.mul(100.0, tf.reduce_mean(tf.cast(correct_pred, tf.float32)))

# Initializing the variables
init = tf.initialize_all_variables()

tf.scalar_summary("loss", cost)
tf.scalar_summary("accuracy", accuracy)
merged_summary_op = tf.merge_all_summaries()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    # Keep training until reach max iterations
    for epoch in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        loss, acc, optimize, summary = sess.run([cost, accuracy, optimizer, merged_summary_op],
                                                feed_dict={input_data: batch_x, output_data: batch_y,
                                                           keep_prob: drop_out})
        summary_writer.add_summary(summary, epoch * batch_size)
        if epoch % display_step == 0:
            print("Iter ", epoch, " Mini batch Loss= ", "{:.3f}".format(
                loss) + ", Training Accuracy= ", "{:.3f}".format(acc))
    print("Optimization Finished!")
    print("Total test data length: ", len(mnist.test.images))
    print("Testing Accuracy :", sess.run(accuracy, feed_dict={input_data: mnist.test.images[:],
                                                              output_data: mnist.test.labels[:],
                                                              keep_prob: 1.}))
