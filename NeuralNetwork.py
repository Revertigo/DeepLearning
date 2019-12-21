import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy as np
from joblib.numpy_pickle_utils import xrange
import matplotlib.pyplot as plt


def predict(X):
    return np.round(X)


def draw_graph(training_epochs, history, ylabel, desc):
    plt.plot(list(range(training_epochs)), history)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(desc)

    plt.show()


if __name__ == "__main__":
    # Reading training set(without test set)
    data = pd.read_csv('resources/two_prediction/features_two.csv', header=None)
    features = 13  # Number of coefficient

    # Retrieve features into matrix, then converting that matrix to array
    x_orig = np.array(data.iloc[:, 0:features].values)  # Batch Gradient Descent - using the whole data set

    # Data labels(zero or one, two labels for two classes), then converting the matrix to array
    y_orig = np.array(data.iloc[:, features:].values)
    labels = 2  # 1 bit for prediction classes

    hl1_size = 19
    hl2_size = 13

    # tf Graph Input
    X = tf.compat.v1.placeholder(tf.float32, [None, features])
    # Since this is a binary classification problem, Y takes only 2 values.
    Y = tf.compat.v1.placeholder(tf.float32, [None, labels])  # only 2 classes - male and female.

    # Set model weights
    # Adding first hidden layer
    W1 = tf.Variable(tf.random.truncated_normal([features, hl1_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hl1_size]))
    z1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)  # Using ReLU as activation function in first hidden layer
    # Adding second hidden layer
    W2 = tf.Variable(tf.random.truncated_normal([hl1_size, hl2_size], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[hl2_size]))
    z2 = tf.nn.leaky_relu(tf.matmul(z1, W2) + b2)

    # Adding original output layer
    #W = tf.Variable(tf.random.truncated_normal([hl1_size, labels], stddev=0.1))
    W = tf.Variable(tf.random.truncated_normal([hl2_size, labels], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[labels]))  # Trainable Variable Bias

    # Using Sigmoid since it's a binary classification
    #pred = tf.nn.sigmoid(tf.matmul(z1, W) + b)
    pred = tf.nn.sigmoid(tf.matmul(z2, W) + b)

    cost = tf.reduce_mean(-(Y * tf.math.log(pred) + (1 - Y) * tf.math.log(1 - pred)))
    alpha = 0.008
    # Gradient Descent Optimizer
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(alpha).minimize(cost)  # train phase

    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Resource management technique
    with tf.compat.v1.Session() as sess:
        # Run the initializer
        sess.run(init)
        # Lists for storing the changing Cost and Accuracy in every Epoch
        cost_history, accuracy_history = [], []

        # Reading test set
        test_data = pd.read_csv('resources/two_prediction/features_test_two.csv', header=None)

        # Retrieve features into matrix
        test_data_x = np.array(test_data.iloc[:, 0:features].values)

        # Data labels actual true(zero or one)
        test_data_y = np.array(test_data.iloc[:, features:].values)

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        training_epochs = 1000  # Training cycles
        for epoch in range(training_epochs):
            # Running the Optimizer(training phase)
            sess.run(optimizer, feed_dict={X: x_orig, Y: y_orig})  # BGD, since we are using the whole training set
            # Calculating cost on current Epoch
            c = sess.run(cost, feed_dict={X: x_orig, Y: y_orig})

            # Storing Cost and Accuracy to the history
            cost_history.append(c)
            accuracy_history.append(accuracy.eval({X: x_orig, Y: y_orig}) * 100)

            # Displaying result on current Epoch
            if epoch % 100 == 0 and epoch != 0:
                print("Epoch " + str(epoch) + " Cost: "
                      + str(cost_history[-1]))
                print("Epoch " + str(epoch) + " Accuracy: "
                      + str(accuracy_history[-1]), "%")

        prediction_array = pred.eval(session=sess, feed_dict={X: test_data_x})
        success = 0
        for i in xrange(len(prediction_array)):
            if predict(prediction_array[i][0]) == test_data_y[i][0]:
                success += 1

        print(success)
        print("Actual accuracy based on test set: " + str((success / len(test_data_x)) * 100.0) + " %")

        # Draw cost graph
        draw_graph(training_epochs, cost_history, 'Cost', 'Decrease in Cost with Epochs')
        # Draw accuracy graph
        draw_graph(training_epochs, accuracy_history, 'Accuracy', 'Increase in Accuracy with Epochs')
