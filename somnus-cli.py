import fire
import numpy as np
from tqdm import tqdm

from utils import load_raw_audio
from utils import create_positive_example
from utils import create_negative_example
from utils import create_silent_example
from preprocess_audio import create_dataset
from models import CnnOneFStride
from models import CnnTradFPool


class SomnusCLI():
    def __init__(self, numpy_seed=1):
        np.random.seed(numpy_seed)

    def augment_audio_dataset(self, duration=1, n_positive=90000, n_bgtalk=180000, n_silent=90000, n_negative=120000):
        """
        Method to create the audio dataset for keyword recognition.

        For each audio clip that is generated the following algorithm is executed:

        1. Select an audio clip of background noise
        2. Randomly increase of decrease the DB of the background noise
        3. Overlay a voice clip where applicable with a random shift to the right
        4. Standardize the volume of the audio clip
        5. Set the frame rate of the resulting audio clip to 16k
        6. Write the audio to a file in processed_audio/

        :param duration: The duration of the audio clips in seconds
        :param n_positive: The number of positive examples
        :param n_bgtalk: The number of examples using negative background speech
        :param n_negative: The number of negative examples
        :param n_silent: The number of examples containing only background noise
        """
        print('Please wait while we load the raw audio files...')
        activates, negatives, backgrounds, background_talking = load_raw_audio(duration)

        print('Augmenting positive audio samples:')
        for i in tqdm(range(n_positive)):
            time_shift = np.random.randint(200)
            create_positive_example(backgrounds[i % len(backgrounds)], activates, time_shift, i)

        print('Augmenting background talk audio samples:')
        for i in tqdm(range(n_bgtalk)):
            time_shift = np.random.randint(600)
            create_negative_example(backgrounds[i % len(backgrounds)], activates[0], background_talking[i % len(background_talking)], time_shift, i)

        print('Augmenting background audio audio samples:')
        for i in tqdm(range(n_silent)):
            create_silent_example(backgrounds[i % len(backgrounds)], activates[0], i)

        print('Augmenting negative audio samples:')
        for i in tqdm(range(n_negative)):
            time_shift = np.random.randint(600)
            create_negative_example(backgrounds[i % len(backgrounds)], activates[0], negatives[i % len(negatives)], time_shift, i+180000)


    def preprocess(self,  n_filters=40, show_progress=True, win_length=400, win_hop=160):
        """
        Preprocess the augmented audio and create a dataset of numpy arrays ready for use with the keyword detection models

        :param n_filters: The number of filters in each frame
        :param show_progress: Boolean option to decide whether to show a progress bar (NOTE: showing progress bar may slow down processing)
        :param win_length: The length of each window in frames
        :param win_hop: the number of frames between the starting frame of each consecutive window.
        """
        data, labels = create_dataset(n_filters, show_progress, win_length, win_hop)

        np.save('preprocessed_data/data.npy', data)
        np.save('preprocessed_data/labels.npy', labels)

    def train(self, model='cnn-one-stride', train_split=0.9, n_epochs=200, weights_file='model_weights.hdf5',
                save_best=False, batch_size=64):
        """
        Trains a small-footprint keyword detection model using augmented WAV files

        :param model: The name of the model we want to train
        :param train_split: How much data should be in the training set. Valid values are [0.0, 1.0]
        :param n_epochs: The number of epochs
        :param weights_file: The name of the file the final weights should be saved to
        :param save_best: Whether or not the model should save the best model throughout the training process
        :param batch_size: The size of each mini batch
        """
        train_data = np.load('preprocessed_data/data.npy')
        train_labels = np.load('preprocessed_data/labels.npy')

        shape = train_data[0].shape
        if model == 'cnn-one-stride':
            model = CnnOneFStride(input_shape=shape)
        elif model == 'cnn-trad-pool':
            model = CnnTradFPool(input_shape=shape)
        else:
            raise ValueError("Model type %s not supported" % model)

        # randomly shuffle the data
        p = np.random.permutation(len(train_data))
        train_data = train_data[p]
        train_labels = train_labels[p]

        split = int(train_split * len(train_data))

        model.train(train_data[:split], train_labels[:split], train_data[split:], train_labels[split:], n_epochs, save_best, batch_size)
        model.save(weights_file)


if __name__ == '__main__':
  fire.Fire(SomnusCLI)