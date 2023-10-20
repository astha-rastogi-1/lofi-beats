from train import flatten_notes
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Activation
from music21 import instrument, note, stream, chord

NOTE_TYPE = {
            "eighth": 0.5,
            "quarter": 1,
            "half": 2,
            "16th": 0.25
        }

def test():
    notes = flatten_notes()
    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
    network_input, normalized_input = prepare_network_input(notes, pitchnames, n_vocab)
    model = create_test_model(normalized_input, n_vocab)
    prediction_output = generate_output_notes(model, network_input, pitchnames, n_vocab)
    convert_to_midi(prediction_output)

def prepare_network_input(notes, pitchnames, n_vocab):
    """Prepare the network input to select a random sequence"""

    note_to_number = dict((note, number) for number, note in enumerate(pitchnames))
    sequence_length=100
    network_input = []

    for i in range(len(notes)-sequence_length):
        sequence_in = notes[i:i+sequence_length]
        network_input.append([note_to_number[char] for char in sequence_in])

    n_patterns = len(network_input)
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input/float(n_vocab)

    return network_input, normalized_input

def create_test_model(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
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

    # Load the weights to each node
    model.load_weights('weights-improvement-373-0.0078-bigger.hdf5')
    return model

def generate_output_notes(model, network_input, pitchnames, n_vocab):

    # Pick a random sequence to start with
    start = np.random.randint(0, len(network_input)-1)

    # Create int to note conversion dictionary
    number_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    selected_pattern = network_input[start]
    prediction_output = []

    # Generate 500 notes:
    for note_index in range(50):
        prediction_input = np.reshape(selected_pattern, (1, len(selected_pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = number_to_note[index]
        prediction_output.append(result)

        selected_pattern.append(index)
        selected_pattern = selected_pattern[1: len(selected_pattern)]
    print("Pred: ", prediction_output)
    return prediction_output

def convert_to_midi(prediction_output):
    """Convert prediction output to midi stream"""
    offset = 0
    output_notes = []

    for element in prediction_output:
        # Determine what type of note the current element will be set to
        curr_type = np.random.choice(list(NOTE_TYPE.keys()), p=[0.05,0.65,0.25,0.05])

        # If element is chord
        if ('.' in element) or element.isdigit():
            notes_in_chord = element.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            
            new_chord = chord.Chord(notes, type=curr_type)
            new_chord.offset = offset
            output_notes.append(new_chord)

        elif str(element).lower()=='r':
            curr_type = 'eighth'
            new_rest = note.Rest(type=curr_type)
            new_rest.offset = offset
            output_notes.append(new_rest)

        else:
            new_note = note.Note(element, type=curr_type)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        
        offset += NOTE_TYPE[curr_type]
    
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output.mid')



test()