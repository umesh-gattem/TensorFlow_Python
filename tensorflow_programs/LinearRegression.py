import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

read_file = pd.read_csv("/home/umesh/my_linear_data.csv")
my_input = []
my_output = []
for index, row in read_file.iterrows():
    input_data = row[0]
    output_data = row[len(row) - 1]
    my_input.append(input_data)
    my_output.append(output_data)

train_data_input = np.array(my_input)
train_data_output = np.array(my_output)

no_of_inputs = train_data_input.shape[0]

input_data = tf.placeholder("float")
output_data = tf.placeholder("float")

weights = tf.Variable(np.random.randn())
bias = tf.Variable(np.random.randn())

learning_rate = 0.01
epochs = 1000

activation = tf.add(tf.mul(input_data, weights), bias)

cost = tf.reduce_sum(tf.pow(activation - output_data, 2)) / (2 * no_of_inputs)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for each_epoch in range(epochs):
        for (x, y) in zip(train_data_input, train_data_output):
            sess.run(train_step, feed_dict={input_data: x, output_data: y})

    print("Training completed!")
    training_cost = sess.run(cost, feed_dict={input_data: train_data_input, output_data: train_data_output})
    print("Training cost=", training_cost, "W=", sess.run(weights), "b=", sess.run(bias), '\n')

    plt.plot(train_data_input, train_data_output, 'ro', label='Original data')
    plt.plot(train_data_input, (sess.run(weights) * train_data_input + sess.run(bias)), label='Fitted line')
    plt.legend()
    plt.show()

    test_X = np.array([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.array([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    cost = tf.reduce_sum(tf.pow(activation - output_data, 2)) / (2 * test_X.shape[0])
    testing_cost = sess.run(cost, feed_dict={input_data: test_X, output_data: test_Y})
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_data_input, sess.run(weights) * train_data_input + sess.run(bias), label='Fitted line')
    plt.legend()
    plt.show()


