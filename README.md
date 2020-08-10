# Somnus

![Build](https://github.com/skuldur/somnus/workflows/build/badge.svg)

Somnus allows you to listen for and detect a specific keyword in a continuous stream of audio data. It uses keyword detection models written in Tensorflow 2.0 to detect instances of the keyword and by using small-footprint models Somnus keeps memory usage low and latency to a minimum.

## Getting started

### Prerequisites

```bash
sudo apt-get install portaudio19-dev python-pyaudio python3-pyaudio
```

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the Somnus package and the CLI

```bash
pip install somnus
```

## Usage

### Somnus

Somnus can be used to listen for an instance of a selected keyword in a continuous stream of audio data from a single channel from a microphone. To find the device index of your microphone run `somnus list_microphones`.

Somnus can handle all the audio interfacing for you so that you only need to initialize Somnus and and call the `listen()` and it will start listening to your microphone until it detects the keyword. Somnus also offers a nonblocking method (`detect_keyword()`) that allows the user to process the audio themselves and only use Somnus to detect a keyword in an audio time series passed to `detect_keyword()` as an argument.

Somnus has the following parameters:

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

#### Example

```python
s = Somnus('./model_weights.hdf5', device_index=1)
activated = s.listen()

if activated:
	do_stuff()
```

### CLI

Somnus comes with a CLI that allows you to generate audio data and train your own keyword detection model. The CLI is implemented using Python-Fire. For each command you can use `-h` or `--help` to get a description of the command and a list of the possible arguments for the command.

To start using the CLI run `somnus configure` to create the configuration for the Somnus CLI. Then the raw data directory must contain three sub-directories:

* `positives/` for audio files containing utterances of the keyword. Must contain at least 1 audio file.
* `negatives/` for audio files containing speech that does not contain utterances of the keyword. Must contain at least 1 audio file.
* `backgrounds/` for audio files that contain background noise. This directory is optional but we recommend adding noise to the training data so that the keyword detector also works in noisy conditions.

The CLI currently supports the following audio types: **wav, mp3, flac, ogg, flv, wma, aac**

#### Configure

```bash
somnus configure
```

Create a configuration file with the absolute paths to the:

* Raw audio data directory
* Directory that should contain the augmented audio files
* Directory that should contain the preprocessed data files

**Note** that the augmented audio files and preprocessed data files can use a lot of space so make sure to put them somewhere with a lot of available space.

#### Augmenting audio

```bash
somnus augment_audio
```

The command to generate an audio dataset takes the raw audio in your raw audio directory as input and generates positive, negative, and silent audio files with varying amounts of background noise. These audio files are written to the augmented audio directory.

The command has the following options: 

**--duration**: The duration of the audio clips in seconds  
**--positive**: The number of positive examples  
**--negative**: The number of negative examples  
**--silent**: The number of examples containing only background noise  

#### Preprocessing and creating the dataset
```bash
somnus preprocess
```

The command to preprocess the augmented audio files. It takes the files stored in the augmented audio directory, normalizes them and stores the output array in the preprocessed data directory.

The command has the following options: 

**--filters**: The number of filters in each frame  
**--show_progress**: Boolean option to decide whether to show a progress bar  
**--split**: The split between train, validation, and test data. The total should add up to 1. E.g. `(0.9, 0.05, 0.05)`  
**--win_length**: The length of each window in seconds  
**--win_hop**: the time between the start of each consecutive window.  

#### Training

```bash
somnus train
```

The command to train a small-footprint keyword model loads the data in `./preprocessed_data/` and uses it to train the keyword model.

The command has the following options:

**--model_name**: The name of the model we want to train  
**--epochs**: The number of epochs  
**--weights_file**: The name of the file the final weights should be saved to  
**--save_best**: Whether or not the model should save the best model throughout the training process  
**--batch_size**: The size of each mini batch  
**--lr**: The initial learning rate  

#### Testing

```bash
somnus test
```

The command to test a trained model on a witheld test dataset.

The command has the following options:

**--model_name**: The name of the model we want to test  
**--weights_file**: The path to the weights file  

#### List microphones

```bash
somnus list_microphones
```

Prints out a list of microphones connected to your device along with their device IDs.

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


