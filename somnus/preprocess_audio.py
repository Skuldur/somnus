import glob
import os

import librosa
import librosa.display
import numpy as np
from tensorflow.keras import utils
from tqdm import tqdm


def create_dataset(base_dir, n_filters, show_progress, win_length, win_hop):
    """
    Create a dataset using the normalized representations of the audio files in base_dir

    Args:
        base_dir (string): The directory containing all augmented audio files
        n_filters (int): The number of filters in each frame
        show_progress (boolean): Boolean option to decide whether to show a progress bar (NOTE: showing progress bar may slow down processing)
        win_length (int): The length of each window in frames
        win_hop (int): the number of frames between the starting frame of each consecutive window.

    Returns:
        data (array): A 4D array of normalized audio files stored in arrays in the shape of (X, n_filters, 1), where X is the
            number of windows.
        labels (array): A 1D array of labels corresponding to an array at the same index in the data array. 
    """
    data = []
    labels = []

    # read total number of files for the progress bar
    _, _, files = next(os.walk(base_dir))
    total_files = len(files)

    if show_progress:
        pbar = tqdm(total=total_files)

    def preprocess_data(path, label):
        for filename in glob.iglob(path):
            y, sr = librosa.load(filename, sr = 16000)
            x = melnormalize(y, n_filters, win_length, win_hop)
            data.append(x)
            labels.append(label)

            if show_progress:
                pbar.update(1)

    preprocess_data(os.path.join(base_dir, 'positive*'), 0)
    preprocess_data(os.path.join(base_dir, 'negative*'), 1)
    preprocess_data(os.path.join(base_dir, 'background*'), 2)

    labels = utils.to_categorical(labels)
    return np.array(data), np.array(labels)
        

def melnormalize(audio_time_series, n_filters, win_length, win_hop):
    """
    Normalize and expand a audio time series.

    Args:
        audio_time_series (array): An audio time series
        n_filters (int): The number of filters in each frame
        win_length (int): The length of each window in frames
        win_hop (int): the number of frames between the starting frame of each consecutive window.

    Returns:
        melnormalized (array): A melnormalized representation of the data stored in audio_time_series
    """
    normalizedy = librosa.util.normalize(audio_time_series)

    stft = librosa.core.stft(normalizedy, n_fft = 512, hop_length=win_hop, win_length=win_length)
    mel = librosa.feature.melspectrogram(S=stft, n_mels=n_filters)
    mellog = np.log(mel + 1e-9)
    melnormalized = librosa.util.normalize(mellog)
    melnormalized = np.expand_dims(melnormalized, axis=-1)
    melnormalized = melnormalized.swapaxes(0,1)

    return melnormalized
