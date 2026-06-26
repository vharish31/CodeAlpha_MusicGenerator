# Music Generation with AI Using LSTM Neural Networks

This project implements an AI-powered music generation system using an **LSTM (Long Short-Term Memory)** Recurrent Neural Network in TensorFlow and Keras. The network learns musical structure, notes, timing, and harmonies from standard MIDI files and generates original compositions.

---

## Project Structure

```text
Music_Generation_AI/
│
├── dataset/                # Create this directory and add MIDI (.mid) files
│   ├── classical_1.mid
│   └── jazz_1.mid
│
├── preprocessed_data/      # Generated during preprocessing
│   ├── notes.pkl           # Saved note list
│   └── note_mapping.json   # Note-to-integer dictionary
│
├── model/                  # Folder where weights are saved
│   └── final_music_model.keras
│
├── output/                 # Generated MIDI output files
│   └── generated_lstm_music_YYYYMMDD_HHMMSS.mid
│
├── preprocess.py           # Extracts notes/chords and builds sequences
├── train.py                # Defines and trains the LSTM model
├── generate_music.py       # Loads weights and generates new compositions
├── requirements.txt        # Python dependency manifest
├── README.md               # Setup and usage guidelines (this file)
└── report/                 # Project documentation and performance graphs
    ├── loss_curve.png
    └── accuracy_curve.png
```

---

## Setup Instructions

### 1. Environment Requirements
- **Python 3.8 to 3.11** is recommended.
- Install the required packages via `pip`:

```bash
pip install -r requirements.txt
```

### 2. Populate the Dataset
1. Create a folder named `dataset` in the project root:
   ```bash
   mkdir dataset
   ```
2. Download and place MIDI files in the `dataset/` directory. You can use standard piano datasets such as:
   - Chopin/Beethoven Classical MIDI files.
   - Real book Jazz transcriptions.

---

## Execution Guide

### Step 1: Preprocess the MIDI files
Run `preprocess.py` to extract notes, chords, and build matching integer-sequence arrays:
```bash
python preprocess.py
```
This produces `preprocessed_data/notes.pkl` and mapping files.

### Step 2: Train the LSTM Model
Train the recurrent neural network on the parsed datasets:
```bash
python train.py
```
This compiles the LSTM network and fits the model. It automatically saves intermediate best weights in the `model/` folder and saves loss and accuracy plots inside `report/`.

### Step 3: Generate New Compositions
Generate original music using temperature-guided sampling:
```bash
python generate_music.py
```
This loads your trained model, takes a random start sequence, and outputs a complete new MIDI composition inside the `output/` directory with a timestamp.

---

## Key Hyperparameters

- **Sequence Length**: 100 notes are parsed at a time to predict the next note, preserving context.
- **LSTM Architecture**: 3 layered recurrent structure (512, 512, 256 cells) with 30% dropout to eliminate overfitting.
- **Creativity Temperature**: Sampling prediction outputs with an adjustable factor:
  - `0.2 - 0.5`: highly cohesive, classical style (low variation).
  - `0.8 - 1.0`: balanced jazz/classical feel with natural improvisation.
  - `> 1.2`: progressive, experimental, high entropy.
