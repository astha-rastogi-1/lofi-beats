# LoFi Beats Generation with LSTM Neural Network

This repository contains Python code for training a Long Short-Term Memory (LSTM) neural network to generate lofi beats. The code utilizes the `music21` library to parse MIDI files and create a dataset of musical notes and chords. The LSTM model is then trained on this dataset to generate music sequences.

## Usage
**Data Preparation:** Ensure that your MIDI files are in the `midi_songs` directory. Each MIDI file should contain the musical content you want to use for training. The code will extract notes, chords, and rests from these files.

**Training:** To train the LSTM neural network, run the following. 
```bash
python train.py
```
It performs the following steps:

1. Flattens the musical content from the MIDI files.
2. Prepares the input and output sequences for the network.
3. Defines the LSTM model.
4. Trains the model and saves the weights of the best-performing model.

The trained model will be saved as `weights-improvement-XX-YY-bigger.hdf5`, where XX is the epoch number and YY is the loss.

**Generating Music:** Once the model is trained, you can use it to generate music sequences using `test.py`. 

## Generating Output
To generate an output.mid file from weights previously generated through training, run the following command
```bash
python test.py
```

## Model Architecture
The LSTM model is defined with the following architecture:

- Three LSTM layers with 512 units each.
- Recurrent dropout of 0.3 for regularization.
- Batch normalization after the first LSTM layer.
- Dropout layers with a rate of 0.3 after the first and second dense layers.
- Two dense layers with 256 and the number of unique musical elements as output.
- ReLU activation for the dense layers.
- Softmax activation for the output layer.
## Training Parameters
The training parameters are as follows:

- Number of epochs: 100 .
- Batch size: 128.
- Loss function: Categorical Crossentropy.
- Optimizer: RMSprop.
