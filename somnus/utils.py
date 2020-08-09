import os
import glob
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.flv', '.wma', '.aac']

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio(base_dir, length=1):
    base = AudioSegment.silent(duration=length * 1000)
    def load_directory(audio_dir, loop=False):
        audio_segments = []
        for path in Path(os.path.join(base_dir, audio_dir)).rglob('*.*'):
            if path.suffix in SUPPORTED_AUDIO_EXTENSIONS:
                audio_format = path.suffix.split('.')[-1]
                segment = AudioSegment.from_file(path.absolute(), format=audio_format).set_channels(1)

                if loop:
                    segment = base.overlay(segment, loop=True)
                audio_segments.append(segment)

        return audio_segments

    activates = load_directory('positives')
    backgrounds = load_directory('backgrounds', loop=True)
    negatives = load_directory('negatives')

    return activates, negatives, backgrounds

def create_positive_example(background, activate, time_shift):
    background_var = np.random.randint(-15, 10)
    background = background + background_var

    background = background.overlay(activate, position = time_shift)

    background = match_target_amplitude(background, -20.0)

    background = background.set_frame_rate(16000)

    return background

def create_negative_example(background, dummy, negative, time_shift):
    background_var = np.random.randint(-15, 10)
    background = background + background_var
    background = background.overlay(dummy - 100)

    if len(negative) - 1000 <= 0:
        random_start = np.random.randint(0, 300)
    else:
        random_start = np.random.randint(0, len(negative) - 1000)

    background = background.overlay(negative[random_start:], position = time_shift)

    background = match_target_amplitude(background, -20.0)

    background = background.set_frame_rate(16000)

    return background

def create_silent_example(background, dummy):
    background_var = np.random.randint(-15, 10)
    background = background + background_var
    background = background.overlay(dummy - 100)

    background = background.set_frame_rate(16000)

    return background
