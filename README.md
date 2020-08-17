# Somnus

![Build](https://github.com/skuldur/somnus/workflows/build/badge.svg)
![PyPI - License](https://img.shields.io/pypi/l/somnus)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/skuldur/somnus)

Somnus offers easy keyword detection for everyone. It allows you to listen for and detect a specific keyword in a continuous stream of audio data. It uses keyword detection models developed by Google and Baidu to detect instances of the keyword and by using these small-footprint models Somnus keeps memory usage and latency to a minimum.

## Getting started

### Prerequisites

#### Linux

```bash
sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio
```

#### Windows 10

You need to install Microsoft C++ Build Tools before you can install Somnus.

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the Somnus package and the CLI

```bash
pip install somnus
```

## Quickstart

Somnus makes it simple to go from raw audio recordings to a working keyword detection model. To get started create a few recordings of yourself saying the keyword and download the datasets in the [Recommended datasets section](#recommended-datasets). Move the files to the raw audio directory you specify by running `somnus configure`. 

Now that you have your raw audio files set up, you can use our default configurations to create a highly effective keyword detection model.

1. Run `somnus augment_audio` to augment the audio files with background noise and create your audio dataset
2. Run `somnus preprocess` to normalize the data stored in the augmented audio files and create a dataset that's been prepared for our keyword detection models
3. Run `somnus train --epochs 10` to train a keyword detection model using the dataset you just created. The resulting model will be saved to `saved_model.h5` in your current working directory.
4. Run `somnus test` to test the accuracy of the model you just trained using a test dataset that was generated by the `preprocess` command.

Now that you have a trained model you can use the Somnus client to detect a keyword using your microphone. First run `somnus list_microphones` to find the device index of your microphone. Then run the following test script using your microphone's device index and verify that the keyword detection is working.

```python
from somnus.somnus import Somnus

s = Somnus(model='./saved_model.h5', device_index=1)
activated = s.listen()

if activated:
	print('You did it!')
else:
	print('Something went wrong!')
```

## Usage

### Somnus

Somnus can be used to listen for an instance of a selected keyword in a continuous stream of audio data from a single channel from a microphone. To find the device index of your microphone run `somnus list_microphones`.

Somnus can handle all the audio interfacing for you so that you only need to initialize Somnus and and call the `listen()` and it will start listening to your microphone until it detects the keyword. Somnus also offers a nonblocking method (`detect_keyword()`) that allows the user to process the audio themselves and only use Somnus to detect a keyword in an audio time series passed to `detect_keyword()` as an argument.

**Parameters**
* **keyword_file_path**: The relative or absolute path to a weights file for the keyword model.
* **model (default: 'cnn-one-stride')**: The name of the model you wish to use.
* **device_index (default: 0)**: The device index of the microphone that Somnus should listen to.
* **threshold (default: 0.5)**: A threshold for how confident Somnus has to be for it to detect the keyword
* **audio_config**: A dictionary containing the configuration specific to the audio time series. It contains the following:
	* **data_shape (default: (101, 40, 1))**: The input shape for the keyword model
	* **sample_duration (default: 1)**: How long the input of the keyword model should be in seconds
	* **n_filters (default: 40)**: The number of filters in each frame
	* **win_length (default: 400)**: The length of each window in frames
	* **win_hop (default: 160)**: the number of frames between the starting frame of each consecutive window.

## CLI

[The Somnus CLI Documentation](cli/README.md)

## Models

Currently Somnus offers the choice between the following models:

| Name           | Original paper                                                                                                                                                          | Description                                                                     | Total parameters | Size |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------|-----------|
| cnn-one-stride | [Convolutional Neural Networks for Small-footprint Keyword Spotting](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43969.pdf) | A frequency strided convolutional model with a stride of 4 and no pooling       | 381k                    | 1.5MB     |
| cnn-trad-pool  | [Convolutional Neural Networks for Small-footprint Keyword Spotting](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43969.pdf) | A keyword detection model with two convolutional layers followed by max pooling | 649k                    | 2.5MB     |
| crnn-time-stride  | [Convolutional Recurrent Neural Networks for Small-Footprint Keyword Spotting](https://arxiv.org/ftp/arxiv/papers/1703/1703.05390.pdf) | A convolutional recurrent network with time striding | 88k                    | 380KB     |

## Recommended datasets

Before you start we highly recommend downloading pre-made datasets for both the negative examples and background noise. For negative examples we recommend the [Librispeech](http://www.openslr.org/12/) dataset. You can pick any of the dev, test, or train datasets. To start with we recommend using the `train-clean-100.tar.gz` dataset and moving on to the larger datasets if needed. For background noise we recommend the [DEMAND](https://asa.scitation.org/doi/abs/10.1121/1.4799597) dataset that you can download from Kaggle [here](https://www.kaggle.com/aanhari/demand-dataset).

Extract the data and move the Librispeech dataset to the raw audio directory and place it in the `negatives/` sub-directory and the DEMAND dataset to the `backgrounds/` sub-directory.

`positives/` will then contain utterances of your keyword in various conditions using multiple different voices and dialects. Additionally, you can add custom negative examples to the `negatives/` sub-directory. We recommend that a majority of these utterances use a microphone similar to the one you will be using in the final product. This is because data gathered from different types of microphones can look completely different, e.g. a model trained on utterances recorded using headset microphone will probably not work well with a far field microphone array.

If your model is intended to be used with many different types of microphones then we recommend gathering positive and negative recordings using as many different microphones as you can.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)


