import tensorflow as tf
from my_utils import rztutils

train_data, train_label, test_data, test_label = rztutils.read_csv('mnist_dataset.csv', split_ratio=90, delimiter=";", label_vector=True)

learning_rate = 0.01
epochs = 2000000
batch_size = 128
batch = 0
display_step = 10

no_of_inputs = 784  # no of input pixels
no_of_outputs = 10  # no of outputs
drop_out = 0.75  # keep probability

input_data = tf.placeholder(tf.float32, [None, 784], name='Input')
output_data = tf.placeholder(tf.float32, [None, 10], name='Output')
keep_prob = tf.placeholder(tf.float32)

weights = {
    'weight1': tf.Variable(tf.random_normal([5, 5, 1, 32])),  # for 1st Conv layer C5-32
    'weight2': tf.Variable(tf.random_normal([5, 5, 32, 64])),  # for 2nd Conv layer C5-64
    'weight3': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),  # for fully connected layer
    'weight4': tf.Variable(tf.random_normal([1024, no_of_outputs]))  # for output layer
}

bias = {
    'bias1': tf.Variable(tf.random_normal([32])),
    'bias2': tf.Variable(tf.random_normal([64])),
    'bias3': tf.Variable(tf.random_normal([1024])),
    'bias4': tf.Variable(tf.random_normal([no_of_outputs]))
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


pred = conv_layer(input_data, weights, bias, keep_prob)

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
    step = 0
    # Keep training until reach max iterations
    for epoch in range(epochs):
        while step < len(train_data) / batch_size:
            batch_x, batch_y = train_data[batch: batch + batch_size], train_label[batch:batch + batch_size]
            batch += batch_size
            sess.run(optimizer, feed_dict={input_data: batch_x, output_data: batch_y,
                                           keep_prob: drop_out})
            if epoch % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={input_data: batch_x,
                                                                  output_data: batch_y,
                                                                  keep_prob: 1.})
                print("Iter ", str(step*batch_size), " Mini batch Loss= ", "{:.3f}".format(
                    loss) + ", Training Accuracy= ", "{:.3f}".format(acc))
                step += 1
    print("Optimization Finished!")
    print("Total test data length: ", len(test_data))
    print("Testing Accuracy :", sess.run(accuracy, feed_dict={input_data: train_data[:1000],
                                                              output_data: train_label[:1000],
                                                              keep_prob: 1.}))
