import fire
import numpy as np
from tqdm import tqdm

from utils import load_raw_audio, create_positive_example, create_negative_example, create_silent_example
from preprocess_audio import create_dataset
from models import get_model


class SomnusCLI():
    def __init__(self, numpy_seed=1):
        np.random.seed(numpy_seed)

    def augment_audio_dataset(self, duration=1, positive=90000, bgtalk=180000, silent=90000, negative=120000):
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
        :param positive: The number of positive examples
        :param bgtalk: The number of examples using negative background speech
        :param negative: The number of negative examples
        :param silent: The number of examples containing only background noise
        """
        print('Please wait while we load the raw audio files...')
        activates, negatives, backgrounds, background_talking = load_raw_audio(duration)

        print('Augmenting positive audio samples:')
        for i in tqdm(range(positive)):
            time_shift = np.random.randint(200)
            create_positive_example(backgrounds[i % len(backgrounds)], activates, time_shift, i)

        print('Augmenting background talk audio samples:')
        for i in tqdm(range(bgtalk)):
            time_shift = np.random.randint(600)
            create_negative_example(backgrounds[i % len(backgrounds)], activates[0], background_talking[i % len(background_talking)], time_shift, i)

        print('Augmenting background audio audio samples:')
        for i in tqdm(range(silent)):
            create_silent_example(backgrounds[i % len(backgrounds)], activates[0], i)

        print('Augmenting negative audio samples:')
        for i in tqdm(range(negative)):
            time_shift = np.random.randint(600)
            create_negative_example(backgrounds[i % len(backgrounds)], activates[0], negatives[i % len(negatives)], time_shift, i+180000)


    def preprocess(self,  filters=40, show_progress=True, split=(0.9, 0.05, 0.05), win_length=400, win_hop=160):
        """
        Preprocess the augmented audio and create a dataset of numpy arrays ready for use with the keyword detection models

        :param filters: The number of filters in each frame
        :param show_progress: Boolean option to decide whether to show a progress bar (NOTE: showing progress bar may slow down processing)
        :param split: How much data should be in the training, validation, and test datasets. Values must add up to 1.
        :param win_length: The length of each window in frames
        :param win_hop: the number of frames between the starting frame of each consecutive window.
        """
        assert sum(split) == 1

        data, labels = create_dataset(filters, show_progress, win_length, win_hop)

        # randomly shuffle the data
        p = np.random.permutation(len(data))
        data = data[p]
        labels = labels[p]

        train_data, val_data, test_data = np.split(data, [int(len(data)*split[0]), int(len(data)*(split[0] + split[1]))])
        train_labels, val_labels, test_labels = np.split(labels, [int(len(labels)*split[0]), int(len(labels)*(split[0] + split[1]))])

        np.save('preprocessed_data/train_data.npy', train_data)
        np.save('preprocessed_data/train_labels.npy', train_labels)
        np.save('preprocessed_data/validation_data.npy', val_data)
        np.save('preprocessed_data/validation_labels.npy', val_labels)
        np.save('preprocessed_data/test_data.npy', test_data)
        np.save('preprocessed_data/test_labels.npy', test_labels)

    def train(self, model_name='cnn-one-stride',  epochs=200, weights_file='model_weights.hdf5',
                save_best=False, batch_size=64):
        """
        Trains a small-footprint keyword detection model using augmented WAV files

        :param model_name: The name of the model we want to train
        :param epochs: The number of epochs
        :param weights_file: The name of the file the final weights should be saved to
        :param save_best: Whether or not the model should save the best model throughout the training process
        :param batch_size: The size of each mini batch
        """
        train_data = np.load('preprocessed_data/train_data.npy')
        train_labels = np.load('preprocessed_data/train_labels.npy')
        val_data = np.load('preprocessed_data/validation_data.npy')
        val_labels = np.load('preprocessed_data/validation_labels.npy')

        shape = train_data[0].shape
        model = get_model(model_name)
        model.train(train_data, train_labels, val_data, val_labels, epochs, save_best, batch_size)
        model.save(weights_file)

    def test_model(self, model_name, weights):
        """
        Tests a trained model against a test dataset

        :param model_name: The name of the model we want to test
        :param weights: The path to the weights file
        """
        data = np.load('preprocessed_data/test_data.npy')
        labels = np.load('preprocessed_data/test_labels.npy')

        model = get_model(model_name)

        model.load(weights)

        wrong = 0
        for idx in tqdm(range(len(data))):
            audio = data[idx]
            label = labels[idx]
            p = model.predict(np.expand_dims(audio, axis=0))

            if np.argmax(p) != np.argmax(label):
                wrong += 1

        percentage = 100*((len(data)-wrong) / len(data))
        print("Testset accuracy is %s percent" % percentage)

if __name__ == '__main__':
  fire.Fire(SomnusCLI)