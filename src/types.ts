export interface NoteEvent {
  pitch: number;      // MIDI pitch number (e.g., 60 for C4)
  time: number;       // Start offset in beats
  duration: number;   // Length in beats
  velocity: number;   // Note keypress velocity (0-127)
}

export type Genre = 'classical' | 'jazz' | 'ambient';

export interface TrainingLog {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
  timestamp: string;
}

export type SynthInstrument = 'piano' | 'synth' | 'rhodes';

export interface CodeFile {
  name: string;
  path: string;
  language: string;
  content: string;
}
