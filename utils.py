import glob
import numpy as np
from pydub import AudioSegment


# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio(length=1):
    activates = []
    backgrounds = []
    negatives = []
    background_talking = []

    # 10 second base
    base = AudioSegment.silent(duration=length * 1000)
    for filename in glob.iglob('./raw_data/positives/*.wav'):
        activate = AudioSegment.from_wav(filename).set_channels(1)
        activates.append(activate)
    for filename in glob.iglob('./raw_data/backgrounds/**/*.wav'):
        background = AudioSegment.from_wav(filename)
        backgrounds.append(base.overlay(background, loop=True))
    for filename in glob.iglob('./raw_data/negatives/*.wav'):
        negative = AudioSegment.from_wav(filename).set_channels(1)
        negatives.append(negative)

    for filename in glob.iglob('./raw_data/background_talking/**/**/*.flac'):
        talking = AudioSegment.from_file(filename, format='flac')
        background_talking.append(talking)

    return activates, negatives, backgrounds, background_talking

def create_positive_example(background, activates, time_shift, i):

    background_var = np.random.randint(-15, 10)
    background = background + background_var

    random_index = np.random.randint(len(activates))
    random_activate = activates[random_index]

    random_activate = random_activate
    background = background.overlay(random_activate, position = time_shift)

    background = match_target_amplitude(background, -20.0)

    background = background.set_frame_rate(16000)

    background.export("processed_audio/positive_%s.wav" % i, format='wav')

def create_negative_example(background, dummy, negative, time_shift, i):
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

    background.export("processed_audio/negative_%s.wav" % i, format='wav')

def create_silent_example(background, dummy, i):
    background_var = np.random.randint(-15, 10)
    background = background + background_var
    background = background.overlay(dummy - 100)

    background = background.set_frame_rate(16000)

    background.export("processed_audio/background_%s.wav" % i, format='wav')

