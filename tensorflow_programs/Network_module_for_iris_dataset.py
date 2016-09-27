import tensorflow as tf
import numpy as np
import pandas as pd

read_file = pd.read_csv("/home/umesh/sample.csv")
my_input = []
my_output = []
for index, row in read_file.iterrows():
    input_data = row[:4]
    output_data = row[4:]
    my_input.append(input_data)
    my_output.append(output_data)
train_data_input = np.array(my_input)
train_data_output = np.array(my_output)
# print(train_data_input)
# print(train_data_output.shape)
# exit()

input_list = []
output_list = []
for (x, y) in zip(train_data_input, train_data_output):
    x = np.array([x])
    y = np.array([y])
    # print(x.shape)
    # print(y.shape)
    # exit()
    input_list.append(x)
    output_list.append(y)

no_of_inputs = len(train_data_input)

input_data = tf.placeholder("float")
output_data = tf.placeholder("float")

weights = tf.Variable(tf.truncated_normal((4, 3), dtype=tf.float32))
bias = tf.Variable(tf.truncated_normal((1, 3), dtype=tf.float32))

learning_rate = 0.01
epochs = 10000

activation = tf.add(tf.matmul(input_data, weights), bias)

cost = tf.reduce_sum(tf.pow(activation - output_data, 2)) / (2 * no_of_inputs)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for each_epoch in range(epochs):
        for (x, y) in zip(input_list, output_list):
            sess.run(train_step, feed_dict={input_data: x, output_data: y})
        if ((each_epoch + 1) % 50) == 0:
            training_cost = sess.run(cost, feed_dict={input_data: train_data_input, output_data: train_data_output})
            print("Epoch", each_epoch+1, ":", training_cost)
    print("Training completed!")
    print("Training cost=", training_cost, "W=", sess.run(weights), "b=", sess.run(bias), '\n')

    test_output = []
    for x in input_list:
        output = sess.run(activation, feed_dict={input_data: x})
        # max_index = output.index(max(output))
        # output = [0] * 3
        # output[max_index] = 1
        test_output.append(output)
    print(test_output)

