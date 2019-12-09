import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from pathlib import Path


def resolve_prediction(dir_name):
    prediction = 0  # 0 for female, 1 for male.
    if (dir_name.startswith('m')):
        prediction = 1

    return prediction


def extract_features(wav_file, dir_name):
    """
    The function extracts the features from the wav_file uses as recording file.
    @:param wav_file: Input file to extract features from
    @:param dir_name: directory name the recording in. If it starts with 'm' then it's a male voice.

    @:returns: The function returns a numpy array of features including the prediction concatenated.
    """
    (rate, sig) = wav.read(wav_file)
    mfcc_feat = mfcc(sig, rate, winfunc=np.hamming)
    # For now, i won't use this feature processing technique.
    # feature_inputs = np.asarray(mfcc_feat[np.newaxis, :])
    # feature_inputs = (feature_inputs - np.mean(feature_inputs)) / np.std(feature_inputs)
    # print(feature_inputs)

    # Write mfcc output features into one vector contains all features.

    # data = np.asarray(mfcc_feat).reshape(-1)  # Convert feature matrix to one big feature array. Not used for now
    data = np.mean(mfcc_feat, axis=0)  # average all columns
    data = np.append(data, int(resolve_prediction(dir_name)))  # add the prediction to the end of the feature array.

    return data


if __name__ == "__main__":
    # path to the training set
    data_set_path = 'C:\\Users\\Dekel\\Downloads\\לימודים\\deep learning\\an4_sphere\\an4\\wav\\an4_clstk\\'
    # path to the test set
    data_set_test_path = 'C:\\Users\\Dekel\\Downloads\\לימודים\\deep learning\\an4_sphere\\an4\\wav\\an4test_clstk\\'

    path_list = Path(data_set_test_path).glob('**/*.wav')
    with open("features_test.csv", 'w') as f:
        for path in path_list:
            path_to_wav = str(path)  # convert object path to string
            dir_name = path_to_wav.rsplit('\\', 2)[1]  # extract directory name
            arr_feat = extract_features(path_to_wav, dir_name)  # Features array
            for i in range(len(arr_feat) - 1):
                f.write("%s," % str(arr_feat[i]))
            f.write("%s\n" % str(arr_feat[len(arr_feat) - 1]))
