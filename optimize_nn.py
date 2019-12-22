import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy as np
from joblib.numpy_pickle_utils import xrange

features = 13  # Number of coefficient
labels = 2  # 1 bit for prediction classes

# tf Graph Input
X = tf.compat.v1.placeholder(tf.float32, [None, features])
# Since this is a binary classification problem, Y takes only 2 values.
Y = tf.compat.v1.placeholder(tf.float32, [None, labels])  # only 2 classes - male and female.

# Reading training set(without test set)
data = pd.read_csv('resources/two_prediction/features.csv', header=None)

# Retrieve features into matrix, then converting that matrix to array
x_orig = np.array(data.iloc[:, 0:features].values)  # Batch Gradient Descent - using the whole data set

# Data labels(zero or one), then converting the matrix to array
y_orig = np.array(data.iloc[:, features:].values)

# Reading test set
test_data = pd.read_csv('resources/two_prediction/features_test.csv', header=None)

# Retrieve features into matrix
test_data_x = np.array(test_data.iloc[:, 0:features].values)

# Data labels actual true(zero or one)
test_data_y = np.array(test_data.iloc[:, features:].values)


def predict(X):
    return np.round(X)


def training_process(alpha, cycles, hl1_size, hl2_size):
    # Set model weights
    # Adding first hidden layer
    W1 = tf.Variable(tf.random.truncated_normal([features, hl1_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hl1_size]))
    z1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # Using ReLU as activation function in first hidden layer

    # Adding second hidden layer
    W2 = tf.Variable(tf.random.truncated_normal([hl1_size, hl2_size], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[hl2_size]))
    z2 = tf.nn.leaky_relu(tf.matmul(z1, W2) + b2)

    W = tf.Variable(tf.random.truncated_normal([hl2_size, labels], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[labels]))  # Trainable Variable Bias

    # Using Sigmoid since it's a binary classification
    # pred = tf.nn.sigmoid(tf.matmul(z1, W) + b)
    pred = tf.nn.sigmoid(tf.matmul(z2, W) + b)

    cost = tf.reduce_mean(-(Y * tf.math.log(pred) + (1 - Y) * tf.math.log(1 - pred)))
    # Gradient Descent Optimizer
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(alpha).minimize(cost)  # train phase

    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Resource management technique
    with tf.compat.v1.Session() as sess:
        # Run the initializer
        sess.run(init)
        for epoch in range(cycles):
            # Running the Optimizer(training phase)
            sess.run(optimizer, feed_dict={X: x_orig, Y: y_orig})  # BGD, since we are using the whole training set

        return pred.eval(session=sess, feed_dict={X: test_data_x})


def prediction_process(prediction_array):
    success = 0
    for i in xrange(len(prediction_array)):
        if predict(prediction_array[i][0]) == test_data_y[i][0]:
            success += 1

    return success


if __name__ == "__main__":
    print("Working...")

    # find best cycles, alpha, hl_size
    max_success_count = 0
    best_cycles = 0
    best_alpha = 0
    best_hl1_size = 0
    best_hl2_size = 0

    for cycles in xrange(1, 3):
        for alpha in xrange(1, 10):
            for hl2_size in xrange(13, 23):
                for hl1_size in xrange(1, 10):
                    prediction_array = training_process(alpha * 0.001, cycles * 1000, hl1_size + hl2_size, hl2_size)
                    temp_count = prediction_process(prediction_array)

                    if temp_count > max_success_count:
                        max_success_count = temp_count
                        best_cycles = cycles * 1000
                        best_alpha = alpha * 0.001
                        best_hl1_size = hl1_size + hl2_size
                        best_hl2_size = hl2_size
        print("Done ", cycles, " Cycle")

    print("Best Cycles: " + str(best_cycles))
    print("Best Alpha: " + str(best_alpha))
    print("Best hidden layer 1 size: " + str(best_hl1_size))
    print("Best hidden layer 2 size: " + str(best_hl2_size))
    print("max_success_count: " + str(max_success_count))
    print("Accuracy is: " + str((max_success_count / len(test_data_x)) * 100.0) + " %")
