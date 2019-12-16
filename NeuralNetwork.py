import tensorflow as tf
import pandas as pd
import numpy as np
from joblib.numpy_pickle_utils import xrange


def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))


def predict(X):
    return np.round(X)


if __name__ == "__main__":
    # Reading training set(without test set)
    data = pd.read_csv('features.csv', header=None)

    # Retrieve features into matrix, then converting that matrix to array
    x_orig = np.array(data.iloc[:, 0:-1].values)  # Batch Gradient Descent - using the whole data set

    # Data labels(zero or one), then converting the matrix to array
    y_orig = np.array(data.iloc[:, -1:].values)

    features = 13  # Number of coefficient
    labels = 1  # 1 bit for prediction classes

    # (hl1_size, hl2_size) = (100, 50)
    hl1_size = 10

    # tf Graph Input
    X = tf.compat.v1.placeholder(tf.float32, [None, features])
    # Since this is a binary classification problem, Y takes only 2 values.
    Y = tf.compat.v1.placeholder(tf.float32, [None, labels])  # only 2 classes - male and female.

    # Set model weights
    # Adding first hidden layer
    W1 = tf.Variable(tf.truncated_normal([features, hl1_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hl1_size]))
    z1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # Using ReLU as activation function in first hideden layer
    # Adding second hidden layer
    # W2 = tf.Variable(tf.truncated_normal([hl1_size, hl2_size], stddev=0.1))
    # b2 = tf.Variable(tf.constant(0.1, shape=[hl2_size]))
    # z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)

    # Adding original output layer
    W = tf.Variable(tf.truncated_normal([hl1_size, labels], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[labels]))  # Trainable Variable Bias

    # Using Sigmoid since it's a binary classification
    pred = tf.nn.sigmoid(tf.matmul(z1, W) + b)

    cross_entropy = tf.reduce_mean(-(Y * tf.log(pred) + (1 - Y) * tf.log(1 - pred)))
    # y_clipped = tf.clip_by_value(Y, 1e-10, 0.9999999)
    # cross_entropy = -tf.reduce_mean(tf.reduce_sum(pred * tf.log(y_clipped)
    #                                               + (1 - pred) * tf.log(1 - y_clipped), axis=1))
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y)  # loss
    alpha = 0.001
    # Gradient Descent Optimizer
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)  # train phase

    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Resource management technique
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        # Reading test set
        test_data = pd.read_csv('features_test.csv', header=None)

        # Retrieve features into matrix
        test_data_x = np.array(test_data.iloc[:, 0:-1].values)

        # Data labels actual true(zero or one)
        test_data_y = np.array(test_data.iloc[:, -1:].values)

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        training_epochs = 5000  # Training cycles
        for epoch in range(training_epochs):
            # Running the Optimizer(training phase)
            sess.run(optimizer, feed_dict={X: x_orig, Y: y_orig})  # BGD, since we are using the whole training set
            #print(epoch," ", sess.run(accuracy, feed_dict={X: test_data_x, Y: test_data_y}))

        prediction_array = pred.eval(session=sess, feed_dict={X: test_data_x})
        print(prediction_array)
        success = 0
        for i in xrange(len(prediction_array)):
            # prediction = predict(np.matmul(np.array(test_data_x[i]), sess.run(W)) + sess.run(b))
            if predict(prediction_array[i]) == test_data_y[i]:
                success += 1

        print(success)
        print("Accuracy is: " + str((success / len(test_data_x)) * 100.0) + " %")
