import os
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
    
    # Start training (default parameters; adjust batch_size/epochs depending on machine specs)
    epochs = 50
    batch_size = 64
    
    print(f"Starting model training for {epochs} epochs (Batch Size: {batch_size})...")
    
    # Split some data for validation
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
    
    # Generate and save the performance plots
    plot_training_results(history)
    print("Training phase completed successfully!")

if __name__ == "__main__":
    train_network()
