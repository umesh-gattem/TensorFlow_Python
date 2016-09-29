import tensorflow as tf
import numpy as np
import pandas as pd

read_file = pd.read_csv("iris_data.csv")
my_input = []
my_output = []
for index, row in read_file.iterrows():
    input_data = row[:4]
    output_data = row[4:]
    my_input.append(input_data)
    my_output.append(output_data)
train_data_input = np.array(my_input)
train_data_output = np.array(my_output)

input_list = []
output_list = []
for (x, y) in zip(train_data_input, train_data_output):
    x = np.array([x])
    y = np.array([y])
    input_list.append(x)
    output_list.append(y)

no_of_inputs = len(train_data_input)
learning_rate = 0.9
epochs = 100
logs_path = '/tmp/tensorflow_logs/my_example'

input_data = tf.placeholder("float", name='Inputdata')
output_data = tf.placeholder("float", name='Outputdata')

weights = tf.Variable(tf.random_normal((4, 3), dtype=tf.float32), name='Weights')
bias = tf.Variable(tf.random_normal((1, 3), dtype=tf.float32), name='Bias')
# count = tf.Variable(20, dtype=tf.float32)


with tf.name_scope('Activation'):
    activation = tf.add(tf.matmul(input_data, weights), bias)

with tf.name_scope('Loss'):
    cost = tf.reduce_sum(tf.pow(activation - output_data, 2)) / (2 * no_of_inputs)

with tf.name_scope('Optimization'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# with tf.name_scope('Accuracy'):
#     print(len(train_data_input))
#     # accuracy = tf.div(count, len(train_data_input))
#     # accuracy = tf.mul(accuracy, 100)
#     acc = tf.equal(pred, actual)
#     acc = tf.reduce_mean(tf.cast(acc, tf.float32))

init = tf.initialize_all_variables()

tf.scalar_summary("loss", cost)
merged_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    for each_epoch in range(epochs):
        for (x, y) in zip(input_list, output_list):
            optimizer, summary = sess.run([train_step, merged_summary_op], feed_dict={input_data: x, output_data: y})
            summary_writer.add_summary(summary, each_epoch * len(train_data_input))

        if ((each_epoch + 1) % 1) == 0:
            training_cost = sess.run(cost, feed_dict={input_data: train_data_input, output_data: train_data_output})
            print("Epoch", each_epoch+1, ":", "cost is :", training_cost)
    print("Training completed!")
    print("Training cost=", training_cost, "W=", sess.run(weights), "b=", sess.run(bias), '\n')

    test_output = []
    for x in input_list:
        print(x)
        output = sess.run(activation, feed_dict={input_data: x})
        index = np.argmax(output)
        print(index)
        result = np.zeros(output.shape, dtype=float)
        result[0, index] = 1
        test_output.append(result)

    count = 0
    print(test_output)      
    for x, y in zip(test_output, output_list):
        # accuracy = acc.eval({pred: x, actual: y})
        if (x == y).all():
            count += 1
    # print(sess.run(count))
    # acc = sess.run(accuracy)
    # print(sess.run(count))
    print(count)
    accuracy = (count / len(train_data_input)) * 100
    print("Accuracy: ", accuracy)



