# Somnus

Somnus allows you to listen for and detect a specific keyword in a continuous stream of audio data. It uses small-footprint keyword detection models written in Tensorflow 2.0 to detect instances of the keyword and by using small-footprint models Somnus keeps memory usage low and latency to a minimum.

## Getting started

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies

```bash
pip install -r requirements.txt
```

### Recommended datasets

Before you start we highly recommend downloading pre-made datasets for both the background talking and background noise. For background talking we recommend the [Librispeech](http://www.openslr.org/12/) dataset. You can pick any of the clean dev, test, or train datasets. To start with we recommend using the `test-clean.tar.gz` dataset and moving on to the larger datasets if needed. For background noise we recommend the [DEMAND](https://asa.scitation.org/doi/abs/10.1121/1.4799597) dataset that you can download from Kaggle [here](https://www.kaggle.com/aanhari/demand-dataset).

Extract the data and move the Librispeech dataset to `raw_data/background_talking` and the DEMAND dataset to `raw_data/backgrounds`.

`raw_data/positives` will then contain utterances of your keyword in various conditions using multiple different voices and dialects and `raw_data/negatives` contains custom utterances that you may want to add to the examples of background talking. We recommend that a majority of these utterances use a microphone similar to the one you will be using in the final product. This is because data gathered from different types of microphones can look completely different, e.g. a model trained on utterances recorded using headset microphone will probably not work well with a far field microphone array.

If your model is intended to be used with many different types of microphones then we recommend gathering positive and negative recordings using as many different microphones as you can.

## Usage

### Somnus

Somnus can be used to listen for an instance of a selected keyword in a continuous stream of audio data from a single channel from a microphone (multi-channel support will be added at a later date).

Somnus can handle all the audio interfacing for you so that you only need to initialize Somnus and and call the `listen()` and it will start listening to your microphone until it detects the keyword. Somnus also offers a nonblocking method (`detect_keyword()`) to that allows the user to process the audio themselves and only use Somnus to detect a keyword in an audio time series passed to `detect_keyword()` as an argument.

Somnus has the following parameters:

* keyword_file_path: The relative or absolute path to a weights file for the keyword model.
* model (default: 'cnn-one-stride'): The name of the model you wish to use.
* device_index (default: 0): The device index of the microphone that Somnus should listen to.
* threshold (default: 0.9): A threshold for how confident Somnus has to be for it to detect the keyword
* data_shape (default: (101 40 1)): The input shape for the keyword model
* sample_duration (default: 1): How long the input of the keyword model should be in seconds
* n_filters (default: 40): The number of filters in each frame
* win_length (default: 400): The length of each window in frames
* win_hop (default: 160): the number of frames between the starting frame of each consecutive window.

#### Example

```python
s = Somnus('./model_weights.hdf5', device_index=1)
activated = s.listen()

if activated:
	do stuff
```

### CLI

Somnus comes with a CLI that allows you to generate audio data and train your own keyword detection model. The CLI is implemented using Python-Fire. For each command you can use `-h` or `--help` to get a description of the command and a list of the possible arguments for the command.

#### Augmenting audio

```bash
python somnus-cli.py augment_audio_dataset
```

The command to generate an audio dataset takes the raw audio in `./raw_data/` as input and generates positive, negative, and silent audio files with varying amounts of background noise. These audio files are written to `./processed_audio/`.

The command has the following options: 

* duration: The duration of the audio clips in seconds
* n_positive: The number of positive examples
* n_bgtalk: The number of examples using negative background speech
* n_negative: The number of negative examples
* n_silent: The number of examples containing only background noise

#### Preprocessing and creating the dataset
```bash
python somnus-cli.py preprocess
```

The command to preprocess the augmented audio files. It takes the files stored in `./processed_audio/` and melnormalizes them and stores the output array in `./preprocessed_data/`.

The command has the following options: 

* n_filters: The number of filters in each frame
* show_progress: Boolean option to decide whether to show a progress bar (NOTE: showing progress bar may slow down processing)
* win_length: The length of each window in frames
* win_hop: the number of frames between the starting frame of each consecutive window.

#### Training

```bash
python somnus-cli.py train
```

The command to train a small-footprint keyword model loads the data in `./preprocessed_data/` and uses it to train the keyword model.

The command has the following options:

* model: The name of the model we want to train
* train_split: How much data should be in the training set. Valid values are [0.0, 1.0]
* n_epochs: The number of epochs
* weights_file: The name of the file the final weights should be saved to
* save_best: Whether or not the model should save the best model throughout the training process
* batch_size: The size of each mini batch

## Models

Currently Somnus offers the choice between the following models:

| Name           | Source                                                                                                                                                          | Description                                                                     | Total parameters | Size |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------|-----------|
| cnn-one-stride | [Convolutional Neural Networks for Small-footprint Keyword Spotting](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43969.pdf) | A frequency strided convolutional model with a stride of 4 and no pooling       | 381k                    | 1.5MB     |
| cnn-trad-pool  | [Convolutional Neural Networks for Small-footprint Keyword Spotting](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43969.pdf) | A keyword detection model with two convolutional layers followed by max pooling | 649k                    | 2.5MB     |

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)


