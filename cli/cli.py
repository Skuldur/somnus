import json
from functools import wraps
from pathlib import Path
from types import FunctionType
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import fire
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm
import pyaudio

from somnus.utils import load_raw_audio, create_positive_example, create_negative_example, create_silent_example

CONFIG_ERROR_MSGS = {
    'raw_audio': 'Raw Audio Directory is not set. ',
    'augmented_audio': 'Augmented Audio Directory is not set. ',
    'preprocessed_data': 'Preprocessed Data Directory is not set. '
}

def config_wrapper(method, config):
    @wraps(method)
    def wrapped(*args, **kwargs):
        for key in CONFIG_ERROR_MSGS.keys():
            if not config.get(key):
                raise ValueError(CONFIG_ERROR_MSGS[key] + 'Please run \'somnus configure\' to set value!')
        
        method(*args, **kwargs)
    return wrapped

class ConfigWrapper(type):
    def __new__(meta, classname, bases, classDict):
        newClassDict = {}
        config = classDict['config']

        # Wraps every public method except configure so that the user needs to run 'somnus configure'
        # before using the other methods
        for attributeName, attribute in classDict.items():
            if isinstance(attribute, FunctionType) and attributeName != 'configure' and attributeName[0] != '_': 
                # replace it with a wrapped version
                attribute = config_wrapper(attribute, config)
            newClassDict[attributeName] = attribute
        return type.__new__(meta, classname, bases, newClassDict)


class SomnusCLI(metaclass=ConfigWrapper):
    config = {}

    def __init__(self, numpy_seed=1, base_dir='.'):
        np.random.seed(numpy_seed)

        self._load_config()
        
    def _load_config(self):
        config = {}
        config_file = os.path.join(Path.home(), '.somnus', 'config.json')
        if os.path.isfile(config_file):
            with open(config_file, 'r') as file:
                config = json.load(file)

        # we copy the values over so that the config wrapper has a reference to the correct config
        for key, val in config.items():
            self.config[key] = val

    def configure(self):
        config = {}
        config['raw_audio'] = input("Raw Audio Directory [%s]: " % self.config.get('raw_audio')) or self.config.get('raw_audio')
        config['augmented_audio'] = input("Augmented Audio Directory [%s]: " % self.config.get('augmented_audio')) or self.config.get('augmented_audio')
        config['preprocessed_data'] = input("Preprocessed Data Directory [%s]: " % self.config.get('preprocessed_data')) or self.config.get('preprocessed_data')

        config_file = os.path.join(Path.home(), '.somnus', 'config.json')
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as file:
            file.write(json.dumps(config))

    def augment_audio(self, duration=1, positive=90000, negative=120000, background=90000):
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
        :param negative: The number of negative examples
        :param background: The number of examples containing only background noise
        """

        print('Please wait while we load the raw audio files...')
        positives, negatives, backgrounds = load_raw_audio(self.config['raw_audio'], duration)

        if len(positives) == 0:
            raise IndexError('There are no supported audio files in %s/%s' % (self.config['raw_audio'], 'positives'))
        if len(negatives) == 0:
            raise IndexError('There are no supported audio files in %s/%s' % (self.config['raw_audio'], 'negatives'))
        if len(backgrounds) == 0:
            # Add a single silent background audio segment
            backgrounds.append(AudioSegment.silent(duration=duration * 1000))

        aug_path = self.config['augmented_audio']
        print('Augmenting positive audio samples:')
        for i in tqdm(range(positive)):
            time_shift = np.random.randint(200)
            segment = create_positive_example(backgrounds[i % len(backgrounds)], positives[i % len(positives)], time_shift)

            segment.export(os.path.join(aug_path, 'positive_%s.wav' % i), format='wav')

        print('Augmenting negative audio samples:')
        for i in tqdm(range(negative)):
            time_shift = np.random.randint(600)
            segment = create_negative_example(backgrounds[i % len(backgrounds)], positives[0], negatives[i % len(negatives)], time_shift)

            segment.export(os.path.join(aug_path, 'negative_%d.wav' % i), format='wav')

        print('Augmenting background audio audio samples:')
        for i in tqdm(range(background)):
            segment = create_silent_example(backgrounds[i % len(backgrounds)], positives[0])

            segment.export(os.path.join(aug_path, 'background_%s.wav' % i), format='wav')

    def preprocess(self,  filters=40, show_progress=True, split=(0.9, 0.05, 0.05), win_length=0.025, win_hop=0.01):
        """
        Preprocess the augmented audio and create a dataset of numpy arrays ready for use with the keyword detection models

        :param filters: The number of filters in each frame
        :param show_progress: Boolean option to decide whether to show a progress bar (NOTE: showing progress bar may slow down processing)
        :param split: How much data should be in the training, validation, and test datasets. Values must add up to 1.
        :param win_length: The length of each window in seconds
        :param win_hop: the time between the start of each consecutive window.
        """
        from somnus.preprocess_audio import create_dataset
        assert sum(split) == 1

        augmented_path = self.config['augmented_audio']
        preprocessed_path = self.config['preprocessed_data']
        frame_rate = 16000

        win_length = int(win_length * frame_rate)
        win_hop = int(win_hop * frame_rate)

        data, labels = create_dataset(augmented_path, filters, show_progress, win_length, win_hop)

        # randomly shuffle the data
        p = np.random.permutation(len(data))
        data = data[p]
        labels = labels[p]

        train_data, val_data, test_data = np.split(data, [int(len(data)*split[0]), int(len(data)*(split[0] + split[1]))])
        train_labels, val_labels, test_labels = np.split(labels, [int(len(labels)*split[0]), int(len(labels)*(split[0] + split[1]))])

        np.save(os.path.join(preprocessed_path, 'train_data.npy'), train_data)
        np.save(os.path.join(preprocessed_path, 'train_labels.npy'), train_labels)
        np.save(os.path.join(preprocessed_path, 'validation_data.npy'), val_data)
        np.save(os.path.join(preprocessed_path, 'validation_labels.npy'), val_labels)
        np.save(os.path.join(preprocessed_path, 'test_data.npy'), test_data)
        np.save(os.path.join(preprocessed_path, 'test_labels.npy'), test_labels)

        audio_config = {
            'data_shape': train_data[0].shape, 
            'sample_duration': (train_data[0].shape[0]-1) / (frame_rate // win_hop),
            'n_filters': filters,
            'win_length': win_length,
            'win_hop': win_hop
        }

        print("\nUse the following config as the audio_config parameter in Somnus when using models trained with this dataset: \n\n%s" 
            % json.dumps(audio_config, indent=2, sort_keys=True))

    def train(self, model_name='cnn-one-stride',  epochs=200, weights_file='model_weights.hdf5',
                save_best=False, batch_size=64, lr=0.0001):
        """
        Trains a small-footprint keyword detection model using augmented WAV files

        :param model_name: The name of the model we want to train
        :param epochs: The number of epochs
        :param weights_file: The name of the file the final weights should be saved to
        :param save_best: Whether or not the model should save the best model throughout the training process
        :param batch_size: The size of each mini batch
        :param lr: The initial learning rate
        """
        from somnus.models import get_model

        preprocessed_path = self.config['preprocessed_data']

        train_data = np.load(os.path.join(preprocessed_path, 'train_data.npy'))
        train_labels = np.load(os.path.join(preprocessed_path, 'train_labels.npy'))
        val_data = np.load(os.path.join(preprocessed_path, 'validation_data.npy'))
        val_labels = np.load(os.path.join(preprocessed_path, 'validation_labels.npy'))

        shape = train_data[0].shape
        model = get_model(model_name, shape)
        model.compile(lr)
        model.train(train_data, train_labels, val_data, val_labels, epochs, save_best, batch_size)
        model.save(weights_file)

    def test(self, model_name='cnn-one-stride', weights='model_weights.hdf5'):
        """
        Tests a trained model against a test dataset

        :param model_name: The name of the model we want to test
        :param weights: The path to the weights file
        """
        from somnus.models import get_model

        preprocessed_path = self.config['preprocessed_data']

        data = np.load(os.path.join(preprocessed_path, 'test_data.npy'))
        labels = np.load(os.path.join(preprocessed_path, 'test_labels.npy'))

        model = get_model(model_name, data[0].shape)

        model.load(weights)

        wrong = 0
        for idx in tqdm(range(len(data))):
            audio = data[idx]
            label = labels[idx]
            p = model.predict(np.expand_dims(audio, axis=0))

            if np.argmax(p) != np.argmax(label):
                wrong += 1

        percentage = 100*((len(data)-wrong) / len(data))
        print("\n Test dataset accuracy is %s percent" % percentage)

    def list_microphones(self):
        """
        List all microphones connected to the device
        """
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


def main():
    fire.Fire(SomnusCLI)
