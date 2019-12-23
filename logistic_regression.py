import pandas as pd
import numpy as np
from joblib.numpy_pickle_utils import xrange


def sigmoid(x, w, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))


def predict(x, w, b):
    return round(sigmoid(x, w, b))


def prediction_process(test_data_x, test_data_y, w, b):
    success = 0

    # Prediction phase
    for i in xrange(len(test_data_x)):
        prediction = predict(test_data_x[i], w, b)
        if test_data_y[i] == prediction:
            success += 1
    return success


def training_process(fx, fy, alpha, cycles):
    # the . in the first element is for creating float array
    w = np.array([0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 13 weights for 13 features
    b = 0

    # Training phase
    for iteration in xrange(cycles):
        gradient_b = np.mean(1 * ((sigmoid(fx, w, b)) - fy))
        gradient_w = np.dot((sigmoid(fx, w, b) - fy), fx) * 1 / len(fy)
        b -= alpha * gradient_b
        w -= alpha * gradient_w

    return w, b


if __name__ == "__main__":
    # Reading training set
    data = pd.read_csv('resources/one_prediction/features.csv', header=None)

    # Retrieve features into matrix, then converting the matrix to array
    data_x = np.squeeze(np.asarray(data.iloc[:, 0:-1].values))

    # Data labels(zero or one), then converting the matrix to array
    data_y = np.squeeze(np.asarray(data.iloc[:, -1:].values))

    fx = np.squeeze(np.asarray(data_x))  # convert to array
    fy = np.squeeze(np.asarray(data_y))  # convert to array

    alpha = 0.109
    cycles = 8000
    w, b = training_process(fx, fy, alpha, cycles)  # alpha = 0.109, cycles = 8000

    # Reading test set
    test_data = pd.read_csv('resources/one_prediction/features_test.csv', header=None)

    # Retrieve features into matrix
    test_data_x = test_data.iloc[:, 0:-1].values

    # Data labels actual true(zero or one)
    test_data_y = test_data.iloc[:, -1:].values

    success = prediction_process(test_data_x, test_data_y, w, b)
    print("Hit count: ", success, "(out of 130)")
    print("Actual accuracy based on test set: " + str((success / len(test_data_x)) * 100.0) + " %")
