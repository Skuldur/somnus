from queue import Queue
from threading import Thread
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pyaudio

from somnus.models import get_model
from somnus.preprocess_audio import melnormalize


class Somnus():
    """
    Args:
        keyword_file_path (string): The relative or absolute path to a weights file for the keyword model.
        model (string): The name of the model you wish to use.
        device_index (int): The device index of the microphone that Somnus should listen to.
        threshold (float): A threshold for how confident Somnus has to be for it to detect the keyword (between [0,1])
        data_shape (tuple): The input shape for the keyword model
        sample_duration (float): How long the input of the keyword model should be in seconds
        n_filters (int): The number of filters in each frame
        win_length (int): The length of each window in frames
        win_hop (int): the number of frames between the starting frame of each consecutive window.
    """
    def __init__(
            self, 
            keyword_file_path='',
            model=None,
            model_name='cnn-one-stride',
            device_index=0, 
            threshold=0.5, 
            audio_config=None
        ):

        if not audio_config:
            audio_config = self._get_default_config()

        if model:
            self.model = model
        else:
            self.model = get_model(model_name, audio_config['data_shape'])
            self.model.load(keyword_file_path)


        self.chunk_duration = 0.1 # Each read length in seconds from mic.
        self.fs = 16000 # sampling rate for mic
        self.chunk_samples = int(self.fs * self.chunk_duration) # Each read length in number of samples.

        # Each model input data duration in seconds, need to be an integer numbers of chunk_duration
        self.feed_samples = int(self.fs * audio_config['sample_duration'])
        
        self.threshold = threshold

        # Data buffer for the input wavform
        self.data = np.zeros(self.feed_samples, dtype='int16')
        self.device_index = device_index

        # variables for preprocessing the audio stream
        self.n_filters = audio_config['n_filters']
        self.win_length = audio_config['win_length']
        self.win_hop = audio_config['win_hop']

        # Optional variables for continuous listening mode
        # Queue to communiate between the audio callback and main thread
        self.q = None
        self.stream = None
        self.listening = False

    def listen(self):
        """
        Fetches data from the audio buffer until it detects a trigger word

        Returns:
            True if the key word is detected, otherwise False
        """
        self._setup_stream()
        try:
            self.stream.start_stream()
            while True:
                audio_stream = self.q.get().astype('float')
                result, confidence = self._get_prediction(audio_stream)

                if result == 0 and confidence > self.threshold:
                    self.listening = False
                    return True           
        except (KeyboardInterrupt, SystemExit):
            self.stream.stop_stream()
            self.stream.close()
            sys.exit()
        except:
            # if something fails then we return False
            return False

    def detect_keyword(self, audio_stream):
        """
        Normalizes the audio_stream argument and detects whether or not it contains the key word

        Args:
            audio_stream (array): An audio time series

        Returns:
            True if the key word is detected, otherwise False
        """
        result, confidence = self._get_prediction(audio_stream)

        if result == 0 and confidence > self.threshold:
                return True
        return False

    def _get_audio_input_stream(self):
        stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.fs,
            input=True,
            frames_per_buffer=self.chunk_samples,
            input_device_index=self.device_index,
            stream_callback=self._callback)
        return stream

    def _get_default_config(self):
        """The default config assumes that all the default arguments for the Somnus CLI were used"""
        return {
            'data_shape': (101, 40, 1), 
            'sample_duration': 1.,
            'n_filters': 40,
            'win_length': 400,
            'win_hop': 160
        }

    def _callback(self, in_data, frame_count, time_info, status):         
        data0 = np.frombuffer(in_data, dtype='int16')
        
        self.data = np.append(self.data,data0)    
        if len(self.data) > self.feed_samples:
            self.data = self.data[-self.feed_samples:]
            # Process data async by sending a queue.
            if self.listening:
                self.q.put(self.data)
        return (in_data, pyaudio.paContinue)

    def _setup_stream(self):
        """ 
        Initialize the audio stream for continuous listening
        """
        self.stream = self._get_audio_input_stream()
        self.listening = True
        self.q = Queue()
        self.data = np.zeros(self.feed_samples, dtype='int16')

    def _get_prediction(self, audio_stream):
        """
        Predicts the class of the audio time series

        Args:
            audio_stream (array): An audio time series

        Returns:
            Returns the predicted class and the confidence the model has in its prediction
        """
        data = melnormalize(audio_stream, self.n_filters, self.win_length, self.win_hop)
        data = np.expand_dims(data, axis=0)

        preds = self.model.predict(data)
        res = np.argmax(preds)

        return res, max(preds)
