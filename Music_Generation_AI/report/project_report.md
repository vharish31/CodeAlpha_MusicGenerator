# ACADEMIC PROJECT REPORT

## Project Title
**Music Generation with AI Using LSTM Neural Networks**

**Course Module:** Artificial Intelligence & Deep Learning  
**Project Classification:** Generative Deep Learning, Recurrent Neural Networks (RNN)  

---

## 1. Introduction
Generative Art and Music represent one of the most exciting frontiers of modern Artificial Intelligence. Traditionally, music composition was deemed a uniquely human intellectual endeavor requiring emotional intelligence and structural reasoning. However, with advancements in Recurrent Neural Networks (RNNs) and specifically **Long Short-Term Memory (LSTM)** architectures, deep learning models can now model long-range temporal dependencies and harmonic intervals. This project focuses on building an autonomous music synthesizer capable of learning compositional techniques from classical and jazz music scores, encoded in MIDI format, and producing original, aesthetic, and harmonically coherent pieces of music.

---

## 2. Problem Statement
The central challenge of symbolic music generation lies in preserving **temporal coherence** across multi-scale intervals. A melody is not merely a randomized sequence of independent note transitions; rather, it possesses:
1. **Short-term structural syntax:** The transition from one note to the next, adhering to local key signatures and scales (chord progressions).
2. **Long-term harmonic coherence:** Themes, motifs, repetitions, and phrases that return multiple bars later, defining the macro-structure of the piece.

Standard Feedforward Neural Networks and traditional Markov Chains fail to capture these long-range temporal hierarchies. Markov Chains are limited by their finite memory order ($k$), causing them to degenerate into randomized, aimless patterns. Classic Recurrent Neural Networks suffer from **vanishing and exploding gradient problems**, making it mathematically impossible to learn connections separated by more than 10-15 time steps. Thus, a robust model with specialized gated memory architectures is required to preserve compositional context.

---

## 3. Project Objectives
The core goals of this project are:
* **Feature Extraction:** Build an automated pipeline using Python's `music21` library to parse multi-track MIDI streams, extract melodic event series (pitches, chords, intervals), and map them into standardized numerical sequence tensors.
* **Architecture Implementation:** Design a deep LSTM recurrent neural network in Keras/Tensorflow capable of learning non-linear, high-dimensional representation spaces of musical sequences.
* **Heuristic Creativity Control:** Implement a Temperature-guided Softmax sampling algorithm to modulate the degree of predictive entropy (allowing the composition style to range from safe, predictable classics to experimental jazz improvisations).
* **Generation & Synthesis:** Reconstruct the numerical output vectors into physical MIDI files containing proper note velocities, offsets, and instrument mappings.

---

## 4. System Architecture
The system follows a sequential pipeline where raw MIDI files are ingested, preprocessed, passed into an LSTM network for training, and subsequently used by a sampling generator to compile new scores:

```text
+-----------------------+      +-----------------------+      +-----------------------+
|  Raw MIDI Dataset     | ---> |   music21 Parser      | ---> | Sequence Builder      |
|  (Classical / Jazz)   |      | (Notes & Chords extraction)  | (Seq Length = 100)    |
+-----------------------+      +-----------------------+      +-----------------------+
                                                                          |
                                                                          v
+-----------------------+      +-----------------------+      +-----------------------+
| New MIDI Output (.mid)| <--- | Temperature Sampling  | <--- | LSTM Deep RNN Model   |
| (Saved in Output/)    |      | (Adjusts randomness)  |      | (Softmax predictions) |
+-----------------------+      +-----------------------+      +-----------------------+
```

---

## 5. Algorithms & Mathematical Framework
The core mathematical driver of this system is the **Long Short-Term Memory (LSTM)** network. An LSTM cell overcomes vanishing gradients by introducing an **internal cell state ($C_t$)** that acts as a continuous conveyor belt, guarded by three non-linear sigmoid gates:

### 5.1 The Forget Gate ($f_t$)
Controls what percentage of the previous cell state $C_{t-1}$ should be discarded.
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

### 5.2 The Input Gate ($i_t$) & Candidate Cell State ($\tilde{C}_t$)
Determines what new information from the current input $x_t$ should be added to the memory.
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

### 5.3 Cell State Update ($C_t$)
Combines the forget gate and input gate to calculate the updated cell state:
$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

### 5.4 The Output Gate ($o_t$) & Hidden State ($h_t$)
Extracts the final output of the LSTM node to be propagated to the next layer.
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * \tanh(C_t)$$

### 5.5 Temperature Sampling Formulation
To introduce controlled unpredictability, the softmax probability scores $P(y_i)$ are scaled by a temperature factor $T$:
$$P_{adj}(y_i) = \frac{\exp(\frac{\ln P(y_i)}{T})}{\sum_j \exp(\frac{\ln P(y_j)}{T})}$$

---

## 6. Dataset Description
For college demonstration, the model leverages MIDI transcriptions of:
1. **Classical Genre:** Compositions by Frédéric Chopin (Nocturnes and Waltzes) and Ludwig van Beethoven (Piano Sonatas). These provide clear mathematical intervals, structural counterpoints, and strict rhythmic timings.
2. **Jazz Genre:** Standard improvisational progressions (such as the ii-V-I turnaround). These are characterized by highly chromatic notes, complex chord inversions, and syncopated timings.

During parsing, note-on and chord events are extracted. Chords are mapped as dot-delimited integers (e.g. `60.64.67` for a C Major Triad), maintaining absolute pitch arrangements.

---

## 7. Implementation Steps
1. **Parsing MIDI:** The `music21.converter` reads file streams.
2. **Constructing Vocabulary:** All unique note/chord labels are aggregated. A vocabulary dictionary `note_to_int` is created.
3. **Sliding Window:** Training slices are created where the previous 100 note events ($X$) are used to predict the 101st note event ($Y$).
4. **Reshaping:** $X$ is reshaped into the 3D tensor shape `(samples, 100, 1)` and scaled to $[0,1]$.
5. **LSTM Training:** The neural network runs for 50 epochs on a GPU workspace, monitoring cross-entropy loss.
6. **Inference Loop:** A seed sequence is selected. The model predicts the next note. The note is sampled, appended to the seed, and the window shifts.

---

## 8. Expected Results
Upon training completion, the cross-entropy loss typically converges from $\sim 5.2$ down to $\sim 1.5$ after 50 epochs, achieving a training accuracy of $65\% - 70\%$. 
- **High Temperature ($T \ge 1.2$):** Produces highly unpredictable, fast-changing sequences that resemble complex avant-garde free jazz.
- **Moderate Temperature ($T = 0.8$):** Balances lyrical predictability with pleasant, unexpected melodic bridges. Excellent for ambient piano solos.
- **Low Temperature ($T \le 0.4$):** Highly repetitive. The model clings to basic scales and often gets stuck in infinite loops, demonstrating the importance of temperature exploration.

---

## 9. Future Enhancements
* **Attention Mechanisms:** Incorporating Transformer Attention layers (similar to Music Transformer) to maintain structural themes over multiple minutes rather than single bars.
* **Multi-Instrument Tracks:** Expanding the LSTM architecture to support multiple channels (e.g. melody, bass line, and drum tracks concurrently).
* **Real-time MIDI Input:** Allowing the model to accept live MIDI input from a keyboard and perform real-time accompaniment.
