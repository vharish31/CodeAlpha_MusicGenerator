import { CodeFile } from '../types';

export const pythonFiles: CodeFile[] = [
  {
    name: "preprocess.py",
    path: "preprocess.py",
    language: "python",
    content: `import glob
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
`
  },
  {
    name: "train.py",
    path: "train.py",
    language: "python",
    content: `import os
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Import sequence loading helper from preprocess
from preprocess import prepare_sequences, SEQUENCE_LENGTH, PREPROCESS_DIR

MODEL_DIR = "model"
REPORT_DIR = "report"

def build_lstm_network(input_shape, n_vocab):
    """
    Constructs and compiles the LSTM Recurrent Neural Network.
    Architecture:
      - LSTM layer with 512 units (returns sequences)
      - Dropout layer (0.3)
      - LSTM layer with 512 units (returns sequences)
      - Dropout layer (0.3)
      - LSTM layer with 256 units
      - Dense layer with 256 units
      - Dropout layer (0.3)
      - Dense Softmax output layer matching the vocabulary size
    """
    model = Sequential()
    
    # First LSTM Layer
    model.add(LSTM(
        512,
        input_shape=(input_shape[1], input_shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    # Second LSTM Layer
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    # Third LSTM Layer
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    # Fully Connected Layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Softmax Output Layer
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    # Compile with Adam and Categorical Crossentropy Loss
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=optimizer, 
        metrics=['accuracy']
    )
    
    return model

def plot_training_results(history):
    """
    Plots and saves loss and accuracy graphs for report submission.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model - Training Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'loss_curve.png'))
    plt.close()
    
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('LSTM Model - Training Accuracy Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORT_DIR, 'accuracy_curve.png'))
    plt.close()
    
    print(f"Training charts saved to directory: '{REPORT_DIR}/'")

def train_network():
    """
    Main training workflow.
    """
    notes_path = os.path.join(PREPROCESS_DIR, 'notes.pkl')
    if not os.path.exists(notes_path):
        print(f"Error: Preprocessed notes not found at {notes_path}.")
        print("Please run preprocess.py first to parse the MIDI files.")
        return

    # Load preprocessed notes list
    with open(notes_path, 'rb') as filepath:
        notes = pickle.load(filepath)
        
    n_vocab = len(set(notes))
    
    # Prepare sequence data
    network_input, network_output, pitchnames = prepare_sequences(notes, n_vocab)
    
    # Convert outputs to categorical one-hot encoding
    network_output_categorical = to_categorical(network_output, num_classes=n_vocab)
    
    # Build LSTM network
    print("Building LSTM Neural Network Model...")
    model = build_lstm_network(network_input.shape, n_vocab)
    model.summary()
    
    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras")
    
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    callbacks_list = [checkpoint, early_stopping]
    
    # Start training
    epochs = 50
    batch_size = 64
    
    print(f"Starting model training for {epochs} epochs...")
    
    history = model.fit(
        network_input, 
        network_output_categorical, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1,
        callbacks=callbacks_list
    )
    
    # Save final fully-trained model weights
    final_model_path = os.path.join(MODEL_DIR, "final_music_model.keras")
    model.save(final_model_path)
    print(f"Saved complete trained model to: {final_model_path}")
    
    # Generate and save performance plots
    plot_training_results(history)
    print("Training phase completed successfully!")

if __name__ == "__main__":
    train_network()
`
  },
  {
    name: "generate_music.py",
    path: "generate_music.py",
    language: "python",
    content: `import os
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
    start_idx = np.random.randint(0, len(network_input) - 1)
    int_to_note = {number: note for number, note in enumerate(pitchnames)}
    
    pattern = network_input[start_idx]
    prediction_output = []

    print(f"Generating {num_notes} notes using temperature: {temperature}...")

    for note_index in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        predictions = model.predict(prediction_input, verbose=0)[0]
        result_index = sample_with_temperature(predictions, temperature)
        result_note = int_to_note[result_index]
        
        prediction_output.append(result_note)

        # Shift sequence window
        pattern = np.append(pattern, result_index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def convert_to_midi(prediction_output, filename="generated_song.mid"):
    """
    Converts the generated list of notes and chords into a standard MIDI file.
    """
    offset = 0
    output_notes = []

    for pattern in prediction_output:
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
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_output_path = os.path.join(OUTPUT_DIR, filename)
    midi_stream.write('midi', fp=full_output_path)
    
    print(f"Generated MIDI file saved successfully to: '{full_output_path}'")
    return full_output_path

def run_generation(num_notes=500, temperature=1.0):
    notes_path = os.path.join(PREPROCESS_DIR, 'notes.pkl')
    mapping_path = os.path.join(PREPROCESS_DIR, 'note_mapping.json')
    model_path = os.path.join(MODEL_DIR, 'final_music_model.keras')
    
    if not (os.path.exists(notes_path) and os.path.exists(mapping_path)):
        print("Error: Dataset mapping not found. Run preprocess.py first.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at '{model_path}'.")
        return

    with open(notes_path, 'rb') as f:
        notes = pickle.load(f)
    
    with open(mapping_path, 'r') as f:
        note_to_int = json.load(f)
        
    pitchnames = sorted(list(note_to_int.keys()))
    n_vocab = len(pitchnames)

    network_input = []
    for i in range(0, len(notes) - SEQUENCE_LENGTH, 1):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[char] for char in sequence_in])

    model = load_model(model_path)
    
    predicted_notes = generate_notes(
        model, 
        network_input, 
        pitchnames, 
        n_vocab, 
        num_notes=num_notes, 
        temperature=temperature
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"generated_lstm_music_{timestamp}.mid"
    convert_to_midi(predicted_notes, filename=file_name)

if __name__ == "__main__":
    run_generation(num_notes=500, temperature=0.8)
`
  },
  {
    name: "requirements.txt",
    path: "requirements.txt",
    language: "text",
    content: `# Python Dependencies for AI LSTM Music Generation
tensorflow>=2.15.0
keras>=2.15.0
numpy>=1.24.3
pandas>=2.0.0
music21>=9.1.0
matplotlib>=3.7.0
midiutil>=1.2.1
`
  },
  {
    name: "README.md",
    path: "README.md",
    language: "markdown",
    content: `# Music Generation with AI Using LSTM Neural Networks

This project implements an AI-powered music generation system using an **LSTM (Long Short-Term Memory)** Recurrent Neural Network in TensorFlow and Keras.

## Setup Instructions

### 1. Environment Requirements
- **Python 3.8 to 3.11** is recommended.
- Install the required packages via \`pip\`:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Populate the Dataset
1. Create a folder named \`dataset\` in the project root.
2. Download and place MIDI files in the \`dataset/\` directory (Classical, Jazz, or Piano tunes).

## Execution Guide

### Step 1: Preprocess
\`\`\`bash
python preprocess.py
\`\`\`

### Step 2: Train Model
\`\`\`bash
python train.py
\`\`\`

### Step 3: Generate Original Melodies
\`\`\`bash
python generate_music.py
\`\`\`
`
  }
];
