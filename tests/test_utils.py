import os
import unittest

from pydub import AudioSegment
import numpy as np

from somnus.utils import load_raw_audio, create_positive_example, create_negative_example, create_silent_example

TEST_DIR = os.path.dirname(os.path.realpath(__file__))


class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_load_raw_audio(self):
        base_dir = os.path.join(TEST_DIR, 'fixtures')
        duration = 1

        pos, neg, back = load_raw_audio(base_dir, duration)

        self.assertEqual(len(pos), 1)
        self.assertEqual(len(neg), 1)
        self.assertEqual(len(back), 1)

    def test_create_positive_example(self):
        base_dir = os.path.join(TEST_DIR, 'fixtures')
        duration = 1

        pos, neg, back = load_raw_audio(base_dir, duration)

        time_shift = np.random.randint(200)
        pos_seg = create_positive_example(back[0], pos[0], time_shift)

        true_seg = AudioSegment.from_wav(os.path.join(base_dir, 'augmented/positive_0.wav')).set_channels(1)

        self.assertSequenceEqual(pos_seg.get_array_of_samples(), true_seg.get_array_of_samples())

    def test_create_negative_example(self):
        base_dir = os.path.join(TEST_DIR, 'fixtures')
        duration = 1

        pos, neg, back = load_raw_audio(base_dir, duration)

        time_shift = np.random.randint(600)
        neg_seg = create_negative_example(back[0], pos[0], neg[0], time_shift)

        true_seg = AudioSegment.from_wav(os.path.join(base_dir, 'augmented/negative_0.wav')).set_channels(1)
        

        self.assertSequenceEqual(neg_seg.get_array_of_samples(), true_seg.get_array_of_samples())

    def test_create_silent_example(self):
        base_dir = os.path.join(TEST_DIR, 'fixtures')
        duration = 1

        pos, neg, back = load_raw_audio(base_dir, duration)

        back_seg = create_silent_example(back[0], pos[0])

        true_seg = AudioSegment.from_wav(os.path.join(base_dir, 'augmented/background_0.wav')).set_channels(1)

        self.assertSequenceEqual(back_seg.get_array_of_samples(), true_seg.get_array_of_samples())






