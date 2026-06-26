import os
import pickle
import numpy as np
import json
import datetime
from music21 import instrument, note, chord, stream
from tensorflow.keras.models import load_model

# Import variables from preprocess
from preprocess import SEQUENCE_LENGTH, PREPROCESS_DIR

MODEL_DIR = "model"
OUTPUT_DIR = "output"

def sample_with_temperature(predictions, temperature=1.0):
    """
    Helper function to sample an index from a probability array using temperature.
    - Low temperature: highly predictable notes (closest to training patterns).
    - High temperature: highly creative and random notes (more adventurous, possibly chaotic).
    """
    if temperature <= 0:
        return np.argmax(predictions)
        
    predictions = np.log(predictions + 1e-10) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

def generate_notes(model, network_input, pitchnames, n_vocab, num_notes=500, temperature=1.0):
    """
    Generates a sequence of notes from the neural network starting with a random seed.
    """
    # Select a random sequence from the input as a starting seed
    start_idx = np.random.randint(0, len(network_input) - 1)
    
    # Int-to-Note reverse mapping
    int_to_note = {number: note for number, note in enumerate(pitchnames)}
    
    pattern = network_input[start_idx]
    prediction_output = []

    print(f"Generating {num_notes} events using a random seed and creativity temperature: {temperature}...")

    for note_index in range(num_notes):
        # Format seed sequence for prediction (1, sequence_length, 1)
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        # Normalize input
        prediction_input = prediction_input / float(n_vocab)

        # Predict probability distribution for the next note
        predictions = model.predict(prediction_input, verbose=0)[0]
        
        # Sample using temperature adjustment
        result_index = sample_with_temperature(predictions, temperature)
        result_note = int_to_note[result_index]
        
        prediction_output.append(result_note)

        # Shift the window: append the predicted index and remove the first element
        pattern = np.append(pattern, result_index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def convert_to_midi(prediction_output, filename="generated_song.mid"):
    """
    Converts the generated list of notes and chords into a standard MIDI file.
    """
    offset = 0
    output_notes = []

    # Reconstruct MIDI notes and chords
    for pattern in prediction_output:
        # If the pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        # If the pattern is a single note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Shift offset by 0.5 beat to prevent overlapping (8th notes style spacing)
        offset += 0.5

    # Build the music21 midi stream
    midi_stream = stream.Stream(output_notes)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Save the MIDI file
    midi_stream.write('midi', fp=full_output_path)
    print(f"Generated MIDI file saved successfully to: '{full_output_path}'")
    return full_output_path

def run_generation(num_notes=500, temperature=1.0):
    """
    Main sequence runner for music generation.
    """
    print("--- Starting Music Generation Stage ---")
    
    # Verify preprocessed metadata exists
    notes_path = os.path.join(PREPROCESS_DIR, 'notes.pkl')
    mapping_path = os.path.join(PREPROCESS_DIR, 'note_mapping.json')
    model_path = os.path.join(MODEL_DIR, 'final_music_model.keras')
    
    if not (os.path.exists(notes_path) and os.path.exists(mapping_path)):
        print("Error: Dataset mapping not found. Run preprocess.py first.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at '{model_path}'.")
        print("Please run train.py first or copy weights to model/final_music_model.keras.")
        return

    # Load mapping configuration
    with open(notes_path, 'rb') as f:
        notes = pickle.load(f)
    
    with open(mapping_path, 'r') as f:
        note_to_int = json.load(f)
        
    pitchnames = sorted(list(note_to_int.keys()))
    n_vocab = len(pitchnames)

    # Reconstruct inputs to find a random sequence seed
    network_input = []
    for i in range(0, len(notes) - SEQUENCE_LENGTH, 1):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[char] for char in sequence_in])

    # Load the trained LSTM model
    print(f"Loading Keras Model from {model_path}...")
    model = load_model(model_path)
    
    # Generate melody
    predicted_notes = generate_notes(
        model, 
        network_input, 
        pitchnames, 
        n_vocab, 
        num_notes=num_notes, 
        temperature=temperature
    )
    
    # Export with custom datetime timestamp to preserve file history
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"generated_lstm_music_{timestamp}.mid"
    
    convert_to_midi(predicted_notes, filename=file_name)

if __name__ == "__main__":
    # Run with standard default config: 500 notes, temperature of 0.8
    run_generation(num_notes=500, temperature=0.8)
