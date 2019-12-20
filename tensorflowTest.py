import tensorflow as tf
import pandas as pd
import numpy as np
from joblib.numpy_pickle_utils import xrange


def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))


def predict(X):
    return np.round(sigmoid(X))


if __name__ == "__main__":

    '''The first step in any automatic speech recognition system is to extract features i.e. identify
    the components of the audio signal that are good for identifying the linguistic content and 
    discarding all the other stuff which carries information like background noise, emotion etc.'''

    '''Computing filter banks and MFCCs involve somewhat the same procedure, where in both cases filter banks are 
    computed and with a few more extra steps MFCCs can be obtained. In a nutshell, a signal goes through a 
    pre-emphasis filter; then gets sliced into (overlapping) frames and a window function is applied to each frame(in 
    our case, hamming window) afterwards, we do a Fourier transform on each frame (or more specifically a Short-Time 
    Fourier Transform) and calculate the power spectrum and subsequently compute the filter banks. To obtain MFCCs, 
    a Discrete Cosine Transform (DCT) is applied to the filter banks retaining a number of the resulting coefficients 
    while the rest are discarded. A final step in both cases, is mean normalization. '''

    '''Steps in a nutshell:
        1. Apply pre-emphasis filter - balance frequency spectrum(reduce it since high frquencies have sammller
            magnitudes compared to lower frequencies.
        2. Framing - chop the signal into short time frames(usually 25ms each). The rationale behind this step 
            is that frequencies in a signal change over time, so in most cases it doesn’
            t make sense to do the Fourier
            transform across the entire signal in that we would lose the frequency contours of the signal over time.
            We also uses a frame step which equals to 10ms which helps in make the frames overlapping.
        3. Window - after slicing the signal into frames, we apply a window function(in our case the Hamming window)
            to each frame.
        4. Fourier-Transform and Power Spectrum - calculate frequency spectrum to each frame and calculate 
            the power spectrum.
        5.  Filter Banks - Compute Filter Banks in order to extract frequency bands.
        6. Mel-frequency Cepstral Coefficients - applying Discrete Cosine Transform (DCT) to decorrelate the 
            filter bank coefficients and yield a compressed representation of the filter banks. 
            For Automatic Speech Recognition, the resulting cepstral coefficients 2-13 are retained and the rest
             are discarded.
    '''
    # Reading training set(without test set)
    data = pd.read_csv('resources/one_prediction/features.csv', header=None)

    # Retrieve features into matrix, then converting that matrix to array
    x_orig = np.array(data.iloc[:, 0:-1].values)

    # Data labels(zero or one), then converting the matrix to array
    y_orig = np.array(data.iloc[:, -1:].values)

    features = 13  # Number of coefficient
    labels = 1  # 1 bit for prediction classes

    # tf Graph Input
    X = tf.compat.v1.placeholder(tf.float32, [None, features])
    # Since this is a binary classification problem, Y take only 2 values.
    Y = tf.compat.v1.placeholder(tf.float32, [None, labels])  # only 2 classes - male and female.

    # Set model weights
    # Trainable Variable Weights
    W = tf.Variable(tf.zeros([features, labels]))
    # Trainable Variable Bias
    b = tf.Variable(tf.zeros([labels]))

    # Using Sigmoid since it's a binary classification
    pred = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))  # Softmax uses for both binary and multi-class classification

    # Sigmoid Cross Entropy Cost Function
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y)  # loss

    alpha = 0.0035

    # Gradient Descent Optimizer
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)  # update

    # Initialize the variables (i.e. assign their default value)
    init = tf.compat.v1.global_variables_initializer()

    # Resource management technique
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        training_epochs = 400 #Training cycles
        for epoch in range(training_epochs):
            # Running the Optimizer(training phase)
            sess.run(optimizer, feed_dict={X: x_orig, Y: y_orig})  # BGD, since we are using the whole training set

        # Reading test set
        test_data = pd.read_csv('resources/one_prediction/features_test.csv', header=None)

        # Retrieve features into matrix
        test_data_x = np.array(test_data.iloc[:, 0:-1].values)

        # Data labels actual true(zero or one)
        test_data_y = np.array(test_data.iloc[:, -1:].values)

        success = 0
        for i in xrange(len(test_data_x)):
            prediction = predict(np.matmul(np.array(test_data_x[i]), sess.run(W)) + sess.run(b))
            if prediction == test_data_y[i]:
                success +=1

        print(success)
        print("Accuracy is: " + str((success / len(test_data_x)) * 100.0) + " %")

