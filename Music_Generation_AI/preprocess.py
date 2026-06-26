import glob
import pickle
import numpy as np
import os
import json
from music21 import converter, instrument, note, chord

# Configuration parameters
SEQUENCE_LENGTH = 100
DATASET_DIR = "dataset"
PREPROCESS_DIR = "preprocessed_data"

def get_notes_from_midi():
    """
    Parses MIDI files from the dataset directory, extracting notes and chords.
    """
    notes = []
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found!")
        print("Please create the directory and populate it with MIDI files (.mid).")
        return notes

    midi_files = glob.glob(os.path.join(DATASET_DIR, "*.mid")) + glob.glob(os.path.join(DATASET_DIR, "*.midi"))
    
    if len(midi_files) == 0:
        print(f"Warning: No MIDI files found in '{DATASET_DIR}'.")
        print("Please place some .mid or .midi files in the dataset folder.")
        return notes

    print(f"Found {len(midi_files)} MIDI files. Starting preprocessing...")

    for file_idx, file in enumerate(midi_files):
        print(f"[{file_idx + 1}/{len(midi_files)}] Parsing {file}...")
        try:
            # Parse the MIDI file
            midi = converter.parse(file)
            
            notes_to_parse = None

            # Get all instruments
            parts = instrument.partitionByInstrument(midi)

            if parts: # If file has multiple instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # If file has a flat note structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Failed to parse {file}: {e}")
            continue

    # Ensure output directories exist
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    
    # Save the parsed notes to a file to avoid reprocessing in the future
    with open(os.path.join(PREPROCESS_DIR, 'notes.pkl'), 'wb') as filepath:
        pickle.dump(notes, filepath)
        
    print(f"Successfully processed {len(notes)} musical events (notes/chords).")
    return notes

def prepare_sequences(notes, n_vocab):
    """
    Prepares the numerical sequences of inputs and outputs required for training the LSTM model.
    """
    # Sort and get unique note pitches/chords
    pitchnames = sorted(list(set(notes)))

    # Create a mapping dictionary from notes to integers
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    
    # Save mapping for use during music generation
    with open(os.path.join(PREPROCESS_DIR, 'note_mapping.json'), 'w') as f:
        json.dump(note_to_int, f, indent=4)

    network_input = []
    network_output = []

    # Create input sequences and corresponding outputs
    for i in range(0, len(notes) - SEQUENCE_LENGTH, 1):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape input to be compatible with LSTM layers (samples, time_steps, features)
    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    
    # Normalize input data to scale between 0 and 1
    network_input = network_input / float(n_vocab)

    # Convert training labels into one-hot vectors
    # (Will be imported dynamically or converted inside train.py)
    network_output = np.array(network_output)

    return network_input, network_output, pitchnames

def run_preprocessing():
    """
    Main entry point for preprocessing workflow.
    """
    print("--- Starting MIDI Preprocessing Stage ---")
    notes = get_notes_from_midi()
    
    if not notes:
        print("No notes parsed. Preprocessing failed.")
        return
        
    n_vocab = len(set(notes))
    print(f"Vocabulary Size (Unique notes/chords): {n_vocab}")
    
    X, y, vocab = prepare_sequences(notes, n_vocab)
    print(f"Input Shape (X): {X.shape}")
    print(f"Output Shape (y): {y.shape}")
    print("Preprocessing completed successfully. Sequences are ready for LSTM training.")

if __name__ == "__main__":
    run_preprocessing()
