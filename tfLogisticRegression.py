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
    plt.plot(list(range(1, training_epochs)), history)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(desc)

    plt.show()


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
    with tf.device('/CPU:0'):
        # Reading training set
        data = pd.read_csv('resources/two_prediction/features.csv', header=None)

        features = 13  # Number of coefficient
        # Retrieve features into matrix, then converting that matrix to array
        x_orig = np.array(data.iloc[:, 0:features].values)

        # Data labels(zero or one), then converting the matrix to array
        y_orig = np.array(data.iloc[:, features:].values)

        labels = 2  # for two classes

        # tf Graph Input
        X = tf.compat.v1.placeholder(tf.float32, [None, features])
        # Since this is a binary classification problem, Y take only 2 values(male and female)
        Y = tf.compat.v1.placeholder(tf.float32, [None, labels])

        # Trainable Variable Weights
        W = tf.Variable(tf.zeros([features, labels]))
        # Trainable Variable Bias
        b = tf.Variable(tf.zeros([labels]))

        # Using Sigmoid since it's a binary classification
        pred = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))  # Softmax uses for both binary and multi-class classification

        # Sigmoid Cross Entropy Cost Function
        cost = tf.reduce_mean(-(Y * tf.math.log(pred) + (1 - Y) * tf.math.log(1 - pred)))
        # alpha = 0.002
        alpha = 0.008

        # Gradient Descent Optimizer
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)  # update

        # Initialize the variables (i.e. assign their default value)
        init = tf.compat.v1.global_variables_initializer()

        # Resource management technique
        with tf.compat.v1.Session() as sess:
            # Run the initializer
            sess.run(init)

            # Lists for storing the changing Cost and Accuracy in every Epoch
            cost_history, accuracy_history = [], []

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            training_epochs = 1000+1  # Training cycles
            for epoch in range(1, training_epochs):
                # Running the Optimizer(training phase)
                sess.run(optimizer, feed_dict={X: x_orig, Y: y_orig})  # BGD, since we are using the whole training set

                # Calculating cost on current Epoch
                c = sess.run(cost, feed_dict={X: x_orig, Y: y_orig})

                # Storing Cost and Accuracy to the history
                cost_history.append(c)
                accuracy_history.append(accuracy.eval({X: x_orig, Y: y_orig}) * 100)

                # Displaying result on current Epoch
                if epoch % 100 == 0:
                    print("Epoch " + str(epoch) + " Cost: "
                          + str(cost_history[-1]))
                    print("Epoch " + str(epoch) + " Accuracy: "
                          + str(accuracy_history[-1]), "%")
            # Reading test set
            test_data = pd.read_csv('resources/two_prediction/features_test.csv', header=None)

            # Retrieve features into matrix
            test_data_x = np.array(test_data.iloc[:, 0:features].values)

            # Data labels actual true(zero or one)
            test_data_y = np.array(test_data.iloc[:, features:].values)

            prediction_array = pred.eval(session=sess, feed_dict={X: test_data_x})
            success = 0
            for i in xrange(len(prediction_array)):
                if predict(prediction_array[i][0]) == test_data_y[i][0]:
                    success += 1

            print("\nHit count: ", success, "(out of 130)")
            print("Actual accuracy based on test set: " + str((success / len(test_data_x)) * 100.0) + " %")

            # Draw cost graph
            draw_graph(training_epochs, cost_history, 'Cost', 'Decrease in Cost with Epochs')
            # Draw accuracy graph
            draw_graph(training_epochs, accuracy_history, 'Accuracy', 'Increase in Accuracy with Epochs')
