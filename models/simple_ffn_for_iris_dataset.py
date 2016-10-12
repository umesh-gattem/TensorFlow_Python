import tensorflow as tf
from utils import read_iris_data_csv_file

train_data_input, train_data_output, test_data_input, test_data_output = read_iris_data_csv_file.read_csv(
    '../datasets/iris_data.csv', split_ratio=70)
no_of_inputs = len(train_data_input)
learning_rate = 0.9
epochs = 1000
display_step = 100
logs_path = 'ffn_for_iris_data'

input_data = tf.placeholder("float", name='Input')
output_data = tf.placeholder("float", name='Output')

weights = {
    'weight1': tf.Variable(tf.random_normal([4, 7], dtype=tf.float32), name='Weight1'),
    'weight2': tf.Variable(tf.random_normal([7, 6], dtype=tf.float32), name='Weight2'),
    'weight3': tf.Variable(tf.random_normal([6, 3], dtype=tf.float32), name='Weight3'),
}

bias = {
    'bias1': tf.Variable(tf.random_normal([7]), dtype=tf.float32, name='Bias1'),
    'bias2': tf.Variable(tf.random_normal([6]), dtype=tf.float32, name='Bias2'),
    'bias3': tf.Variable(tf.random_normal([3]), dtype=tf.float32, name='Bias3')
}


def model(x, weights, bias):
    layer1 = tf.sigmoid(tf.add(tf.matmul(x, weights['weight1']), bias['bias1']))
    layer2 = tf.sigmoid(tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2']))
    output_layer = tf.sigmoid(tf.add(tf.matmul(layer2, weights['weight3']), bias['bias3']))
    return output_layer


with tf.name_scope('Activation'):
    activation = model(input_data, weights, bias)

with tf.name_scope('Loss'):
    cost = tf.reduce_sum(tf.pow(activation - output_data, 2)) / (2 * no_of_inputs)

with tf.name_scope('Optimization'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(activation, 1), tf.argmax(output_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

tf.scalar_summary("loss", cost)
tf.scalar_summary("accuracy", accuracy)
merged_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    for each_epoch in range(epochs):
        training_cost, optimizer, summary = sess.run([cost, train_step, merged_summary_op],
                                                     feed_dict={input_data: train_data_input,
                                                                output_data: train_data_output})
        summary_writer.add_summary(summary, each_epoch * len(train_data_input))

        if each_epoch % display_step == 0:
            print("Epoch", each_epoch, ":", "cost is :", training_cost)
    print("Training cost=", training_cost)

    acc = sess.run(accuracy, feed_dict={input_data: test_data_input, output_data: test_data_output})
    print("Accuracy is :", acc)

