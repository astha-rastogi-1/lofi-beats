# IMPORTS 
import glob
import numpy as np
from music21 import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint

def train_network():
    notes = flatten_notes()
    n_vocab = len(set(notes))
    network_input, network_output = create_sequence(notes, n_vocab)
    model = create_model(network_input, n_vocab)
    train(model, network_input, network_output)

def flatten_notes():
    """Get all the notes from the midi_songs folder and flatten them"""
    # Create an empty list to store flattened notes
    notes = []

    # Iterate through each midi file and flatten the notes within it
    for file in glob.glob(".\midi_songs\*.mid"):
        # print(f"File name: {file}")

        midi = converter.parse(file)
        
        # Get a music21 stream
        notes_to_parse = midi.flat.notes

        # Get the notes, chords and rests within the stream
        # NOTE: Seems like there are only instances of chords and notes in the data
        for element in notes_to_parse:
            # print("INSTANCE: ", element)
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest):
                notes.append('r')
    return notes

def create_sequence(notes, n_vocab):
    """Prepare the sequence used by the Neural Network"""
    
    sequence_length = 100

    # Get list of sortedu unique pitchnames
    pitchnames = sorted(set(item for item in notes))
    # print("PICHNAMES: ", pitchnames)

    # Assign all the pitchnames a number
    note_to_number = dict((note, number) for number, note in enumerate(pitchnames))
    # print("Dict: ", note_to_number)

    network_input = []
    network_output = []

    # Create the ip and op for the network
    for i in range(len(notes) - sequence_length):
        input_sequence = notes[i:i+sequence_length]
        output_note = notes[i+sequence_length]
        network_input.append([note_to_number[char] for char in input_sequence])
        network_output.append(note_to_number[output_note])
    
    num_patterns = len(network_input)

    # print(f"Network Input: {network_input}")
    # print(f"Network Output: {network_output}")

    # Reshaping input for LSTM to a matrix of num_sequences*seqence_length*1 size
    network_input = np.reshape(network_input, (num_patterns, sequence_length, 1))
    # Normalize input:
    network_input = network_input / float(n_vocab)
    # print(f"Network Input: {network_input}")

    # One hot encoding the output so we can use categorical cross entropy during training
    network_output = to_categorical(network_output)

    return network_input, network_output

def create_model(network_input, n_vocab):
    """Model definition"""
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]),
                   recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """Train the neural network"""
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=100, batch_size=128, callbacks=callbacks_list)

if __name__=="__main__":
    train_network()