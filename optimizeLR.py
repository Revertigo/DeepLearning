import pandas as pd
import numpy as np
from joblib.numpy_pickle_utils import xrange
from SelfWrittenLogisticRegression import training_process
from SelfWrittenLogisticRegression import prediction_process

if __name__ == "__main__":
    print("Working...")
    # Reading training set
    data = pd.read_csv('features.csv', header=None)

    # Retrieve features into matrix, then converting the matrix to array
    data_x = np.squeeze(np.asarray(data.iloc[:, 0:-1].values))

    # Data labels(zero or one), then converting the matrix to array
    data_y = np.squeeze(np.asarray(data.iloc[:, -1:].values))

    fx = np.squeeze(np.asarray(data_x))  # convert to array
    fy = np.squeeze(np.asarray(data_y))  # convert to array

    # Reading test set
    test_data = pd.read_csv('features_test.csv', header=None)

    # Retrieve features into matrix
    test_data_x = test_data.iloc[:, 0:-1].values

    # Data labels actual true(zero or one)
    test_data_y = test_data.iloc[:, -1:].values

    # find best cycles, alpha
    max_success_count = 0
    best_cycles = 0
    best_alpha = 0

    for cycles in xrange(1, 15):
        for alpha in xrange(1, 200):
            w, b = training_process(fx, fy, alpha * 0.001, cycles * 1000)
            temp_count = prediction_process(test_data_x, test_data_y, w, b)

            if temp_count > max_success_count:
                max_success_count = temp_count
                best_cycles = cycles * 1000
                best_alpha = alpha * 0.001

    print("Best Cycles: " + str(best_cycles))
    print("Best Alpha: " + str(best_alpha))
    print("max_success_count: " + str(max_success_count))
    print("Accuracy is: " + str((max_success_count / len(test_data_x)) * 100.0) + " %")
