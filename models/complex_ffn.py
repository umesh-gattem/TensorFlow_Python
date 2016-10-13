import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../tmp/data/", one_hot=True)

no_of_inputs = 784
no_of_outputs = 10
learning_rate = 0.9
epochs = 1000
display_step = 100
batch_size = 10

input = tf.placeholder(tf.float32, [None, 784], name='Input')
output = tf.placeholder(tf.float32, [None, 10], name='Output')

weights = {
    'weight1': tf.Variable(tf.random_normal([784, 1024])),
    'weight2': tf.Variable(tf.random_normal([1024, 512])),
    'weight3': tf.Variable(tf.random_normal([512, 784])),
    'weight10': tf.Variable(tf.random_normal([392, 512])),
    'weight11': tf.Variable(tf.random_normal([196, 512])),
    'weight12': tf.Variable(tf.random_normal([512, 10]))
}

bias = {
    'bias1': tf.Variable(tf.random_normal([1024])),
    'bias2': tf.Variable(tf.random_normal([512])),
    'bias3': tf.Variable(tf.random_normal([784])),
    'bias10': tf.Variable(tf.random_normal([512])),
    'bias11': tf.Variable(tf.random_normal([512])),
    'bias12': tf.Variable(tf.random_normal([10]))
}


def model(x, weights, bias):
    layer1 = x
    layer1 = tf.sigmoid(tf.add(tf.matmul(layer1, weights['weight1']), bias['bias1']))
    layer2 = tf.sigmoid(tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2']))
    layer3 = tf.sigmoid(tf.add(tf.matmul(layer2, weights['weight3']), bias['bias3']))
    layer4, layer5, layer6, layer7 = tf.split(1, 4, layer3)
    layer8 = tf.concat(1, [layer4, layer5])
    layer9 = tf.add(layer6, layer7)
    layer10 = tf.sigmoid(tf.add(tf.matmul(layer8, weights['weight10']), bias['bias10']))
    layer11 = tf.sigmoid(tf.add(tf.matmul(layer9, weights['weight11']), bias['bias11']))
    layer12 = tf.sub(layer10, layer11)
    return tf.sigmoid(tf.add(tf.matmul(layer12, weights['weight12']), bias['bias12']))


activation = model(input, weights, bias)
cost = tf.reduce_sum(tf.pow(activation - output, 2)) / (2 * no_of_inputs)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(activation, 1), tf.argmax(output, 1))
accuracy = tf.mul(100.0, tf.reduce_mean(tf.cast(correct_pred, tf.float32)))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for each_epoch in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        loss, optimize, acc = sess.run([cost, optimizer, accuracy], feed_dict={input: batch_x, output: batch_y})

        if each_epoch % display_step == 0:
            print("Iter ", each_epoch, " Training Cost = ", "{:.3f}".format(
                loss) + ", Training Accuracy= ", "{:.3f}".format(acc))

    acc = sess.run(accuracy, feed_dict={input: mnist.test.images[:], output: mnist.test.labels[:]})
    print("Accuracy is :", acc)

