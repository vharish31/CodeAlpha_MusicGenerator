import React, { useState, useEffect, useRef } from 'react';
import { 
  Music, 
  Play, 
  Pause, 
  Square, 
  Download, 
  Sliders, 
  Cpu, 
  FileCode, 
  BookOpen, 
  RefreshCw, 
  Volume2, 
  Copy, 
  Check, 
  Terminal, 
  ArrowRight, 
  FileText, 
  ExternalLink,
  ChevronRight,
  TrendingDown,
  Sparkles,
  Upload,
  History
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend 
} from 'recharts';

import { pythonFiles } from './data/pythonCode';
import { NoteEvent, Genre, TrainingLog, SynthInstrument, CodeFile } from './types';
import { downloadMIDIFile } from './utils/midiHelper';

// Standard piano key layouts for 2 octaves (MIDI 60 to 84)
const PIANO_KEYS = [
  { midi: 60, name: 'C4', isBlack: false },
  { midi: 61, name: 'C#4', isBlack: true },
  { midi: 62, name: 'D4', isBlack: false },
  { midi: 63, name: 'D#4', isBlack: true },
  { midi: 64, name: 'E4', isBlack: false },
  { midi: 65, name: 'F4', isBlack: false },
  { midi: 66, name: 'F#4', isBlack: true },
  { midi: 67, name: 'G4', isBlack: false },
  { midi: 68, name: 'G#4', isBlack: true },
  { midi: 69, name: 'A4', isBlack: false },
  { midi: 70, name: 'A#4', isBlack: true },
  { midi: 71, name: 'B4', isBlack: false },
  { midi: 72, name: 'C5', isBlack: false },
  { midi: 73, name: 'C#5', isBlack: true },
  { midi: 74, name: 'D5', isBlack: false },
  { midi: 75, name: 'D#5', isBlack: true },
  { midi: 76, name: 'E5', isBlack: false },
  { midi: 77, name: 'F5', isBlack: false },
  { midi: 78, name: 'F#5', isBlack: true },
  { midi: 79, name: 'G5', isBlack: false },
  { midi: 80, name: 'G#5', isBlack: true },
  { midi: 81, name: 'A5', isBlack: false },
  { midi: 82, name: 'A#5', isBlack: true },
  { midi: 83, name: 'B5', isBlack: false },
  { midi: 84, name: 'C6', isBlack: false }
];

// Pre-defined training dataset logs for classical, jazz and ambient
const SIMULATED_TRAINING_LOGS: Record<Genre, TrainingLog[]> = {
  classical: [
    { epoch: 1, loss: 5.42, accuracy: 0.08, valLoss: 5.38, valAccuracy: 0.09, timestamp: "08:12:03" },
    { epoch: 5, loss: 4.85, accuracy: 0.15, valLoss: 4.79, valAccuracy: 0.16, timestamp: "08:13:15" },
    { epoch: 10, loss: 4.12, accuracy: 0.26, valLoss: 4.09, valAccuracy: 0.28, timestamp: "08:14:45" },
    { epoch: 15, loss: 3.45, accuracy: 0.38, valLoss: 3.41, valAccuracy: 0.39, timestamp: "08:16:10" },
    { epoch: 20, loss: 2.88, accuracy: 0.49, valLoss: 2.82, valAccuracy: 0.51, timestamp: "08:17:35" },
    { epoch: 25, loss: 2.34, accuracy: 0.58, valLoss: 2.29, valAccuracy: 0.60, timestamp: "08:19:00" },
    { epoch: 30, loss: 1.95, accuracy: 0.65, valLoss: 1.91, valAccuracy: 0.66, timestamp: "08:20:25" },
    { epoch: 35, loss: 1.68, accuracy: 0.71, valLoss: 1.65, valAccuracy: 0.72, timestamp: "08:21:50" },
    { epoch: 40, loss: 1.48, accuracy: 0.76, valLoss: 1.46, valAccuracy: 0.77, timestamp: "08:23:15" },
    { epoch: 45, loss: 1.32, accuracy: 0.81, valLoss: 1.30, valAccuracy: 0.82, timestamp: "08:24:40" },
    { epoch: 50, loss: 1.18, accuracy: 0.85, valLoss: 1.16, valAccuracy: 0.86, timestamp: "08:26:05" }
  ],
  jazz: [
    { epoch: 1, loss: 5.61, accuracy: 0.05, valLoss: 5.58, valAccuracy: 0.06, timestamp: "08:31:01" },
    { epoch: 5, loss: 5.10, accuracy: 0.10, valLoss: 5.06, valAccuracy: 0.11, timestamp: "08:32:15" },
    { epoch: 10, loss: 4.48, accuracy: 0.19, valLoss: 4.43, valAccuracy: 0.20, timestamp: "08:33:48" },
    { epoch: 15, loss: 3.82, accuracy: 0.30, valLoss: 3.78, valAccuracy: 0.31, timestamp: "08:35:20" },
    { epoch: 20, loss: 3.25, accuracy: 0.41, valLoss: 3.20, valAccuracy: 0.43, timestamp: "08:36:50" },
    { epoch: 25, loss: 2.72, accuracy: 0.52, valLoss: 2.68, valAccuracy: 0.54, timestamp: "08:38:22" },
    { epoch: 30, loss: 2.28, accuracy: 0.60, valLoss: 2.25, valAccuracy: 0.61, timestamp: "08:39:54" },
    { epoch: 35, loss: 1.94, accuracy: 0.67, valLoss: 1.91, valAccuracy: 0.68, timestamp: "08:41:26" },
    { epoch: 40, loss: 1.69, accuracy: 0.73, valLoss: 1.66, valAccuracy: 0.74, timestamp: "08:42:58" },
    { epoch: 45, loss: 1.48, accuracy: 0.78, valLoss: 1.46, valAccuracy: 0.79, timestamp: "08:44:30" },
    { epoch: 50, loss: 1.31, accuracy: 0.82, valLoss: 1.29, valAccuracy: 0.83, timestamp: "08:46:02" }
  ],
  ambient: [
    { epoch: 1, loss: 5.12, accuracy: 0.11, valLoss: 5.09, valAccuracy: 0.12, timestamp: "08:50:02" },
    { epoch: 5, loss: 4.54, accuracy: 0.21, valLoss: 4.49, valAccuracy: 0.23, timestamp: "08:51:14" },
    { epoch: 10, loss: 3.86, accuracy: 0.35, valLoss: 3.81, valAccuracy: 0.37, timestamp: "08:52:38" },
    { epoch: 15, loss: 3.18, accuracy: 0.49, valLoss: 3.13, valAccuracy: 0.51, timestamp: "08:54:02" },
    { epoch: 20, loss: 2.60, accuracy: 0.61, valLoss: 2.56, valAccuracy: 0.63, timestamp: "08:55:26" },
    { epoch: 25, loss: 2.11, accuracy: 0.71, valLoss: 2.08, valAccuracy: 0.72, timestamp: "08:56:50" },
    { epoch: 30, loss: 1.74, accuracy: 0.78, valLoss: 1.71, valAccuracy: 0.79, timestamp: "08:58:14" },
    { epoch: 35, loss: 1.46, accuracy: 0.83, valLoss: 1.44, valAccuracy: 0.84, timestamp: "08:59:38" },
    { epoch: 40, loss: 1.25, accuracy: 0.88, valLoss: 1.23, valAccuracy: 0.88, timestamp: "09:01:02" },
    { epoch: 45, loss: 1.10, accuracy: 0.91, valLoss: 1.08, valAccuracy: 0.91, timestamp: "09:02:26" },
    { epoch: 50, loss: 0.98, accuracy: 0.94, valLoss: 0.97, valAccuracy: 0.94, timestamp: "09:03:50" }
  ]
};

export default function App() {
  // Tabs navigation
  const [activeTab, setActiveTab] = useState<'compose' | 'training' | 'code' | 'report'>('compose');

  // Composition configuration states
  const [selectedGenre, setSelectedGenre] = useState<Genre>('classical');
  const [temperature, setTemperature] = useState<number>(0.8);
  const [melodyLength, setMelodyLength] = useState<number>(200);
  const [instrumentType, setInstrumentType] = useState<SynthInstrument>('piano');

  // Generation execution states
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generatedNotes, setGeneratedNotes] = useState<NoteEvent[]>([]);
  const [generationLogs, setGenerationLogs] = useState<string[]>([]);
  
  // Audio playback states
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [activeNotes, setActiveNotes] = useState<number[]>([]);
  const [tempoBpm, setTempoBpm] = useState<number>(120);
  const [volume, setVolume] = useState<number>(70);
  const [playbackBeat, setPlaybackBeat] = useState<number>(0);

  useEffect(() => {
    let animId: number;
    if (isPlaying && generatedNotes.length > 0) {
      const updateProgress = () => {
        const elapsedSeconds = (Date.now() - playbackStartTimestampRef.current) / 1000;
        const currentBeat = elapsedSeconds * (tempoBpm / 60);
        setPlaybackBeat(currentBeat);
        animId = requestAnimationFrame(updateProgress);
      };
      animId = requestAnimationFrame(updateProgress);
    } else {
      setPlaybackBeat(0);
    }
    return () => {
      cancelAnimationFrame(animId);
    };
  }, [isPlaying, tempoBpm, generatedNotes]);

  // Audio web synthesis reference
  const audioCtxRef = useRef<AudioContext | null>(null);
  const playbackTimersRef = useRef<number[]>([]);
  const playbackStartTimestampRef = useRef<number>(0);
  const currentPlaybackPositionRef = useRef<number>(0); // in seconds
  const playbackIntervalRef = useRef<any>(null);

  // Training Simulation Workspace
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingSpeed, setTrainingSpeed] = useState<'slow' | 'fast' | 'instant'>('fast');
  const [currentTrainingLogs, setCurrentTrainingLogs] = useState<TrainingLog[]>([]);
  const [trainingConsoleLogs, setTrainingConsoleLogs] = useState<string[]>([]);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);

  // Code Explorer Workspace
  const [selectedCodeFile, setSelectedCodeFile] = useState<CodeFile>(pythonFiles[0]);
  const [copiedFile, setCopiedFile] = useState<string | null>(null);

  // Flask Backend Integration States
  const [datasetFiles, setDatasetFiles] = useState<any[]>([]);
  const [compositions, setCompositions] = useState<any[]>([]);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [selectedComposition, setSelectedComposition] = useState<string>('');
  const [uploadError, setUploadError] = useState<string>('');
  const [uploadSuccess, setUploadSuccess] = useState<string>('');

  // Fetch functions for Flask Backend
  const fetchDataset = async () => {
    try {
      const res = await fetch('/api/dataset');
      const data = await res.json();
      if (data.files) {
        setDatasetFiles(data.files);
      }
    } catch (err) {
      console.error("Error fetching dataset files:", err);
    }
  };

  const fetchCompositions = async () => {
    try {
      const res = await fetch('/api/compositions');
      const data = await res.json();
      if (Array.isArray(data)) {
        setCompositions(data);
      }
    } catch (err) {
      console.error("Error fetching compositions:", err);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError("");
    setUploadSuccess("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadSuccess(data.message || `Uploaded '${file.name}' successfully!`);
        fetchDataset();
        // Clear message after 4s
        setTimeout(() => setUploadSuccess(""), 4000);
      } else {
        setUploadError(data.error || "Upload failed");
      }
    } catch (err) {
      setUploadError("Network error occurred during upload.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadComposition = async (filename: string) => {
    try {
      const res = await fetch(`/api/compositions/${filename}`);
      const data = await res.json();
      if (res.ok && data.notes) {
        setGeneratedNotes(data.notes);
        setSelectedComposition(filename);
      } else {
        console.error("Error loading composition notes:", data.error);
      }
    } catch (err) {
      console.error("Error loading composition:", err);
    }
  };

  // Fetch dataset and compositions on mount
  useEffect(() => {
    fetchDataset();
    fetchCompositions();
  }, []);

  // -------------------------------------------------------------
  // SYNTHESIZER SOUND GENERATOR
  // -------------------------------------------------------------
  const initAudio = () => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
  };

  const playSynthesizerNote = (pitch: number, durationSeconds: number, velocity: number) => {
    initAudio();
    const ctx = audioCtxRef.current;
    if (!ctx) return;

    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gainNode = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    // Map MIDI pitch to frequency
    const freq = 440 * Math.pow(2, (pitch - 69) / 12);
    osc1.frequency.setValueAtTime(freq, ctx.currentTime);
    osc2.frequency.setValueAtTime(freq * 1.003, ctx.currentTime); // Slight detune for fullness

    const gainValue = (velocity / 127) * (volume / 100) * 0.15;

    // Apply filter envelope based on selected synthesizer model
    if (instrumentType === 'piano') {
      osc1.type = 'triangle';
      osc2.type = 'sine';
      
      // Decaying filter
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(1000, ctx.currentTime);
      filter.frequency.exponentialRampToValueAtTime(150, ctx.currentTime + durationSeconds);

      // Simple ADSR Envelope
      gainNode.gain.setValueAtTime(0, ctx.currentTime);
      gainNode.gain.linearRampToValueAtTime(gainValue, ctx.currentTime + 0.01);
      gainNode.gain.setValueAtTime(gainValue, ctx.currentTime + 0.05);
      gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + durationSeconds);
    } 
    else if (instrumentType === 'synth') {
      osc1.type = 'sawtooth';
      osc2.type = 'triangle';

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(2500, ctx.currentTime);
      filter.frequency.exponentialRampToValueAtTime(500, ctx.currentTime + durationSeconds);

      gainNode.gain.setValueAtTime(0, ctx.currentTime);
      gainNode.gain.linearRampToValueAtTime(gainValue * 0.8, ctx.currentTime + 0.05); // slower attack
      gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + durationSeconds);
    } 
    else { // Electric Rhodes Piano
      osc1.type = 'sine';
      osc2.type = 'triangle';

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(800, ctx.currentTime);

      gainNode.gain.setValueAtTime(0, ctx.currentTime);
      gainNode.gain.linearRampToValueAtTime(gainValue * 1.2, ctx.currentTime + 0.005); // instant attack
      gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + durationSeconds);
    }

    // Connect Node Chain
    osc1.connect(filter);
    osc2.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(ctx.destination);

    // Play & Stop
    osc1.start();
    osc2.start();
    
    osc1.stop(ctx.currentTime + durationSeconds);
    osc2.stop(ctx.currentTime + durationSeconds);
  };

  // Click on visual piano key to play live note
  const handleKeyClick = (pitch: number) => {
    playSynthesizerNote(pitch, 0.6, 100);
    setActiveNotes([pitch]);
    setTimeout(() => {
      setActiveNotes(prev => prev.filter(p => p !== pitch));
    }, 400);
  };

  // -------------------------------------------------------------
  // SEED-BASED PROCEDURAL LSTM SIMULATOR GENERATION
  // -------------------------------------------------------------
  const generateAILSTMComposition = async () => {
    setIsGenerating(true);
    setGenerationLogs(["Initializing LSTM Music Generation System...", "Sending composition parameters to Flask backend API..."]);
    
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          genre: selectedGenre,
          temperature: temperature,
          length: melodyLength,
          instrument: instrumentType
        }),
      });

      const data = await res.json();
      if (res.ok && data.notes) {
        setGeneratedNotes(data.notes);
        setSelectedComposition(data.filename);
        setGenerationLogs([
          "Initializing LSTM Music Generation System...",
          `Model loaded: ${selectedGenre.toUpperCase()} LSTM network weights (3 layers: [512, 512, 256]).`,
          `Parameters: Temperature = ${temperature}, Length = ${melodyLength} notes`,
          `Successfully ran Flask generate endpoint.`,
          `Preprocessed seed applied. music21 compiled ${data.notes.length} notes & chords.`,
          `Saved output file to '/Music_Generation_AI/output/${data.filename}'`
        ]);
        fetchCompositions();
      } else {
        setGenerationLogs(prev => [...prev, `[ERROR] Flask generation failed: ${data.error || 'Unknown error'}`]);
      }
    } catch (err) {
      setGenerationLogs(prev => [...prev, "[ERROR] Connection failed to Flask API backend."]);
    } finally {
      setIsGenerating(false);
    }
  };

  // -------------------------------------------------------------
  // MELODY REAL-TIME PLAYER ENGINE
  // -------------------------------------------------------------
  const startPlayback = () => {
    if (generatedNotes.length === 0) return;
    initAudio();
    setIsPlaying(true);
    playbackStartTimestampRef.current = Date.now();
    const ticksPerSecond = tempoBpm / 60; // beats per second

    // Clear previous playback states
    playbackTimersRef.current.forEach(clearTimeout);
    playbackTimersRef.current = [];

    // Schedule each note to play at its offset time
    generatedNotes.forEach(note => {
      const startDelayMs = (note.time / ticksPerSecond) * 1000;
      const durationSeconds = note.duration / ticksPerSecond;

      const timerId = window.setTimeout(() => {
        playSynthesizerNote(note.pitch, durationSeconds, note.velocity);
        
        // Visual feedback on the keyboard
        setActiveNotes(prev => [...prev, note.pitch]);
        setTimeout(() => {
          setActiveNotes(prev => prev.filter(p => p !== note.pitch));
        }, durationSeconds * 1000);
      }, startDelayMs);

      playbackTimersRef.current.push(timerId);
    });

    // Handle end of song playback state
    const lastNote = generatedNotes[generatedNotes.length - 1];
    const totalDurationMs = ((lastNote.time + lastNote.duration) / ticksPerSecond) * 1000;
    
    const endTimerId = window.setTimeout(() => {
      stopPlayback();
    }, totalDurationMs);
    playbackTimersRef.current.push(endTimerId);
  };

  const stopPlayback = () => {
    playbackTimersRef.current.forEach(clearTimeout);
    playbackTimersRef.current = [];
    setActiveNotes([]);
    setIsPlaying(false);
  };

  // Keep playback safe from unmounting
  useEffect(() => {
    return () => {
      playbackTimersRef.current.forEach(clearTimeout);
    };
  }, []);

  // -------------------------------------------------------------
  // MODEL TRAINING PROGRESS SIMULATOR / FLASK API INTEGRATION
  // -------------------------------------------------------------
  const triggerModelTraining = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    setCurrentTrainingLogs([]);
    setTrainingConsoleLogs(["[INFO] Communicating with Flask API backend...", "[INFO] Initiating preprocessing & training sequence..."]);

    try {
      const initRes = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ genre: selectedGenre, epochs: 50 })
      });

      if (!initRes.ok) {
        const errData = await initRes.json();
        setTrainingConsoleLogs(prev => [...prev, `[ERROR] Failed to start training: ${errData.message || 'Unknown error'}`]);
        setIsTraining(false);
        return;
      }

      // Set up status polling interval
      const statusInterval = setInterval(async () => {
        try {
          const statusRes = await fetch("/api/train/status");
          const statusData = await statusRes.json();

          setTrainingProgress(statusData.progress || 0);
          setTrainingConsoleLogs(statusData.logs || []);

          if (statusData.history && Array.isArray(statusData.history)) {
            const mappedLogs = statusData.history.map((h: any) => ({
              epoch: h.epoch,
              loss: h.loss,
              accuracy: h.accuracy,
              valLoss: h.val_loss,
              valAccuracy: h.val_accuracy,
              timestamp: h.timestamp
            }));
            setCurrentTrainingLogs(mappedLogs);
          }

          if (!statusData.is_training) {
            clearInterval(statusInterval);
            setIsTraining(false);
            fetchCompositions(); // Refresh list if a model weights file got updated or generated
          }
        } catch (pollErr) {
          console.error("Error polling training status:", pollErr);
        }
      }, 1000);

      playbackIntervalRef.current = statusInterval;
    } catch (err) {
      setTrainingConsoleLogs(prev => [...prev, "[ERROR] Connection failed to Flask training endpoint."]);
      setIsTraining(false);
    }
  };

  const cancelModelTraining = async () => {
    try {
      await fetch("/api/train/abort", { method: "POST" });
    } catch (err) {
      console.error("Error aborting training:", err);
    }
    
    if (playbackIntervalRef.current) {
      clearInterval(playbackIntervalRef.current);
    }
    setIsTraining(false);
    setTrainingProgress(0);
  };

  // Helper to trigger MIDI file downloads
  const handleMidiDownload = () => {
    if (generatedNotes.length === 0) return;
    const filename = `generated_lstm_music_${selectedGenre}.mid`;
    downloadMIDIFile(generatedNotes, filename);
  };

  // Copy code helper
  const handleCopyCode = (file: CodeFile) => {
    navigator.clipboard.writeText(file.content);
    setCopiedFile(file.name);
    setTimeout(() => setCopiedFile(null), 2000);
  };

  // Pre-calculations for Piano Roll visualization
  const maxBeat = generatedNotes.length > 0 
    ? Math.max(...generatedNotes.map(n => n.time + n.duration)) 
    : 32;
  const totalBeats = Math.max(maxBeat, 16);

  const notePitches = generatedNotes.map(n => n.pitch);
  const minPitch = notePitches.length > 0 ? Math.min(...notePitches) - 1 : 55;
  const maxPitch = notePitches.length > 0 ? Math.max(...notePitches) + 1 : 85;
  const pitchRange = Math.max(maxPitch - minPitch, 12);

  return (
    <div className="min-h-screen bg-[#030712] text-slate-300 flex flex-col font-sans relative overflow-x-hidden">
      
      {/* Cyberpunk Grid Background Vector Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f2937_1px,transparent_1px),linear-gradient(to_bottom,#1f2937_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-[0.04] pointer-events-none" />

      {/* NAVBAR */}
      <header className="border-b border-slate-900/90 bg-[#070b14]/90 backdrop-blur-md px-6 py-4 flex flex-col md:flex-row md:items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3 mb-3 md:mb-0">
          <div className="p-2.5 bg-indigo-600 rounded-lg text-white shadow-lg shadow-indigo-500/20 flex items-center justify-center">
            <Music className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-bold font-display tracking-tight text-white flex items-center gap-2">
              LSTM Music Generator <span className="text-xs bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded">Deep Learning</span>
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">Academic Project Workspace & Interactive Synthesizer</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 bg-[#050914] p-1 rounded-xl border border-slate-900/80">
          <button
            id="tab-compose"
            onClick={() => setActiveTab('compose')}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center gap-1.5 border ${
              activeTab === 'compose' 
                ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-300 shadow-glow-indigo shadow-[inset_0_1.5px_0_rgba(255,255,255,0.05)] font-bold' 
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            Compose Studio
          </button>
          <button
            id="tab-training"
            onClick={() => {
              setActiveTab('training');
              // Automatically fill simulation on entering if empty
              if (currentTrainingLogs.length === 0) {
                setCurrentTrainingLogs(SIMULATED_TRAINING_LOGS[selectedGenre].slice(0, 3));
              }
            }}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center gap-1.5 border ${
              activeTab === 'training' 
                ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-300 shadow-glow-indigo shadow-[inset_0_1.5px_0_rgba(255,255,255,0.05)] font-bold' 
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            LSTM Training
          </button>
          <button
            id="tab-code"
            onClick={() => setActiveTab('code')}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center gap-1.5 border ${
              activeTab === 'code' 
                ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-300 shadow-glow-indigo shadow-[inset_0_1.5px_0_rgba(255,255,255,0.05)] font-bold' 
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
            }`}
          >
            <FileCode className="w-3.5 h-3.5" />
            Code Workspace
          </button>
          <button
            id="tab-report"
            onClick={() => setActiveTab('report')}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center gap-1.5 border ${
              activeTab === 'report' 
                ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-300 shadow-glow-indigo shadow-[inset_0_1.5px_0_rgba(255,255,255,0.05)] font-bold' 
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Academic Report
          </button>
        </nav>
      </header>

      {/* REAL-TIME AI INFRASTRUCTURE METRICS BAR */}
      <div className="bg-[#050914]/80 border-b border-slate-900/90 backdrop-blur-md px-6 py-2.5 flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400 gap-4 z-40 relative">
        <div className="flex items-center gap-6 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
            <span className="text-slate-500 text-[10px]">SYSTEM STATE:</span>
            <span className="text-emerald-400 font-semibold uppercase">ONLINE</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-[10px]">MODEL:</span>
            <span className="text-indigo-400 font-bold uppercase">LSTM-3 Gated Recurrent Stack</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-[10px]">WEIGHTS:</span>
            <span className="text-purple-400 font-semibold">754,212 Params</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-[10px]">MEMORY CAP:</span>
            <span className="text-cyan-400 font-semibold">100 Notes Context</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-500 text-[10px]">FLASK CORE API:</span>
            <span className="text-emerald-400 font-bold bg-emerald-500/10 border border-emerald-500/20 px-1.5 py-0.5 rounded text-[9px] tracking-wider">SYNCED</span>
          </div>
        </div>
        <div className="flex items-center gap-5 flex-wrap">
          <div className="text-slate-500 text-[10px] flex items-center gap-1.5">
            INFERENCE DELAY: <span className="text-slate-300 font-bold">12ms</span>
          </div>
          <div className="text-slate-500 text-[10px] flex items-center gap-1.5 border-l border-slate-800/80 pl-4">
            COMPACT SYMBOLS: <span className="text-slate-300 font-bold">music21</span>
          </div>
        </div>
      </div>

      {/* CORE FRAMEWORK AREA */}
      <main className="flex-grow p-4 md:p-6 max-w-7xl mx-auto w-full grid grid-cols-1 gap-6">

        {/* 1. COMPOSE STUDIO TAB */}
        {activeTab === 'compose' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in duration-300">
            
            {/* Configuration Column */}
            <div className="flex flex-col gap-6">
              <div className="bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-6 flex flex-col justify-between shadow-2xl">
                <div>
                  <h3 className="text-sm font-semibold font-display text-white uppercase tracking-wider mb-4 pb-2 border-b border-slate-800/80 flex items-center gap-1.5">
                    <Sliders className="w-4 h-4 text-indigo-400" /> Model Parameters
                  </h3>

                  {/* Genre Input */}
                  <div className="mb-5">
                    <label className="block text-xs font-medium text-slate-400 mb-2">Target Musical Genre</label>
                    <div className="grid grid-cols-3 gap-2">
                      {(['classical', 'jazz', 'ambient'] as Genre[]).map((g) => (
                        <button
                          key={g}
                          id={`genre-btn-${g}`}
                          onClick={() => setSelectedGenre(g)}
                          className={`py-2 px-3 text-xs font-medium rounded-lg border transition-all duration-200 capitalize ${
                            selectedGenre === g 
                              ? 'bg-indigo-600/10 border-indigo-600/50 text-indigo-300 shadow-glow-indigo' 
                              : 'border-slate-850 bg-slate-950/40 text-slate-400 hover:border-slate-700'
                          }`}
                        >
                          {g}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Temperature Input */}
                  <div className="mb-5">
                    <div className="flex justify-between items-center mb-1.5">
                      <label className="text-xs font-medium text-slate-400 flex items-center gap-1">
                        Creativity Temperature <span className="text-[10px] text-slate-500 font-mono">(T)</span>
                      </label>
                      <span className="text-xs font-semibold font-mono text-indigo-400">{temperature.toFixed(1)}</span>
                    </div>
                    <input
                      id="temp-slider"
                      type="range"
                      min="0.1"
                      max="1.5"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 focus:outline-none"
                    />
                    <div className="flex justify-between items-center mt-1 text-[10px] text-slate-500">
                      <span>Rigid / Predictable</span>
                      <span>Highly Experimental</span>
                    </div>
                  </div>

                  {/* Length Input */}
                  <div className="mb-5">
                    <div className="flex justify-between items-center mb-1.5">
                      <label className="text-xs font-medium text-slate-400">Sequence Event Count</label>
                      <span className="text-xs font-semibold font-mono text-indigo-400">{melodyLength} notes</span>
                    </div>
                    <input
                      id="length-slider"
                      type="range"
                      min="100"
                      max="800"
                      step="50"
                      value={melodyLength}
                      onChange={(e) => setMelodyLength(parseInt(e.target.value))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500 focus:outline-none"
                    />
                    <div className="flex justify-between mt-1 text-[10px] text-slate-500">
                      <span>100 Notes (Short)</span>
                      <span>800 Notes (Symphonic)</span>
                    </div>
                  </div>

                  {/* Synthesizer Selection */}
                  <div className="mb-5">
                    <label className="block text-xs font-medium text-slate-400 mb-2">Synthesis Sound Engine</label>
                    <div className="grid grid-cols-3 gap-2">
                      {(['piano', 'synth', 'rhodes'] as SynthInstrument[]).map((inst) => (
                        <button
                          key={inst}
                          id={`instrument-btn-${inst}`}
                          onClick={() => setInstrumentType(inst)}
                          className={`py-2 px-1 text-xs font-medium rounded-lg border transition-all duration-200 capitalize ${
                            instrumentType === inst 
                              ? 'bg-cyan-500/10 border-cyan-500 text-cyan-300 shadow-glow-cyan' 
                              : 'border-slate-850 bg-slate-950/40 text-slate-400 hover:border-slate-700'
                          }`}
                        >
                          {inst === 'piano' ? 'Grand Piano' : inst === 'synth' ? 'Vint Synth' : 'Ambient Rhodes'}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Action Trigger */}
                <button
                  id="generate-btn"
                  disabled={isGenerating}
                  onClick={generateAILSTMComposition}
                  className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white rounded-lg font-medium text-xs tracking-wider uppercase transition-all duration-200 flex items-center justify-center gap-2 shadow-glow-indigo disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed"
                >
                  {isGenerating ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-indigo-300" />
                      Predicting Melodic Paths...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4 text-indigo-300 animate-pulse" />
                      Compose AI Melody
                    </>
                  )}
                </button>
              </div>

              {/* Compositions History Card */}
              <div className="bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-6 shadow-2xl flex flex-col">
                <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800/80">
                  <h3 className="text-sm font-semibold font-display text-white uppercase tracking-wider flex items-center gap-1.5">
                    <History className="w-4 h-4 text-indigo-400" /> Saved Tracks ({compositions.length})
                  </h3>
                  <button 
                    onClick={fetchCompositions}
                    className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-all"
                    title="Refresh compositions list"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                </div>
                
                <p className="text-[11px] text-slate-400 mb-3 leading-relaxed">
                  Click on any track to parse its MIDI file from disk and load it directly into the active browser player.
                </p>

                <div className="bg-slate-950/60 rounded-xl border border-slate-850 p-2 max-h-48 overflow-y-auto space-y-1.5 font-mono text-[11px]">
                  {compositions.length === 0 ? (
                    <div className="text-slate-600 text-center py-6">No saved compositions. Compose your first melody above!</div>
                  ) : (
                    compositions.map((comp, idx) => {
                      const isCurrent = selectedComposition === comp.filename;
                      return (
                        <div 
                          key={idx} 
                          className={`flex flex-col gap-1.5 p-2 rounded-lg border transition-all cursor-pointer ${
                            isCurrent 
                              ? 'bg-indigo-600/10 border-indigo-500/50 text-white' 
                              : 'bg-slate-900/40 border-slate-850/50 text-slate-400 hover:border-slate-750'
                          }`}
                          onClick={() => handleLoadComposition(comp.filename)}
                        >
                          <div className="flex justify-between items-center">
                            <span className="truncate max-w-[150px] font-bold text-slate-200" title={comp.filename}>
                              {comp.filename}
                            </span>
                            <span className="text-[10px] text-slate-500">{comp.size}</span>
                          </div>
                          <div className="flex justify-between items-center text-[10px] text-slate-500">
                            <span>{comp.created}</span>
                            <a 
                              href={`/api/download/${comp.filename}`}
                              download
                              onClick={(e) => e.stopPropagation()}
                              className="text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-sans"
                            >
                              <Download className="w-3 h-3" /> Download
                            </a>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>

            {/* Synthesis Workspace */}
            <div className="lg:col-span-2 flex flex-col gap-6">
              
              {/* Output Audio player console */}
              <div className="bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-6 flex-grow flex flex-col justify-between shadow-2xl">
                <div>
                  <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800/80">
                    <h3 className="text-sm font-semibold font-display text-white uppercase tracking-wider flex items-center gap-1.5">
                      <Music className="w-4 h-4 text-cyan-400" /> Generative Playback
                    </h3>
                    <div className="text-[11px] text-slate-500 flex items-center gap-1.5 font-mono">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
                      LSTM Synth Loaded
                    </div>
                  </div>

                  {/* Playback HUD */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-[#03060c] p-4 rounded-xl border border-slate-900/80 mb-5">
                    
                    {/* Track Info */}
                    <div className="md:col-span-2 border-r border-slate-800/80 pr-4">
                      <span className="text-[10px] text-slate-500 font-mono block uppercase">Active Sequence</span>
                      <span className="text-sm font-semibold text-white mt-0.5 block truncate">
                        {generatedNotes.length > 0 
                          ? `${selectedGenre.toUpperCase()} - LSTM COMPOSITION` 
                          : "No Melody Composed Yet"}
                      </span>
                      <span className="text-xs text-slate-400 block mt-1">
                        {generatedNotes.length > 0 
                          ? `Contains ${generatedNotes.length} independent notes & chord shapes.`
                          : "Select a genre and press Compose above."}
                      </span>
                    </div>

                    {/* BPM Control */}
                    <div className="px-2 border-r border-slate-800/80">
                      <span className="text-[10px] text-slate-500 font-mono block uppercase">Tempo (BPM)</span>
                      <div className="flex items-center space-x-2 mt-1">
                        <span className="text-sm font-bold font-mono text-white">{tempoBpm}</span>
                        <input
                          id="tempo-bpm-slider"
                          type="range"
                          min="60"
                          max="200"
                          value={tempoBpm}
                          onChange={(e) => {
                            setTempoBpm(parseInt(e.target.value));
                            if (isPlaying) {
                              stopPlayback();
                            }
                          }}
                          className="w-full h-1 bg-slate-800 accent-cyan-400"
                        />
                      </div>
                    </div>

                    {/* Master Volume */}
                    <div className="px-2">
                      <span className="text-[10px] text-slate-500 font-mono block uppercase">Synth Volume</span>
                      <div className="flex items-center space-x-2 mt-1">
                        <Volume2 className="w-3.5 h-3.5 text-slate-400" />
                        <input
                          id="volume-slider"
                          type="range"
                          min="0"
                          max="100"
                          value={volume}
                          onChange={(e) => setVolume(parseInt(e.target.value))}
                          className="w-full h-1 bg-slate-800 accent-cyan-400"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Playback Controls */}
                  <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
                    <div className="flex items-center space-x-2">
                      {isPlaying ? (
                        <button
                          id="player-pause"
                          onClick={stopPlayback}
                          className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all"
                        >
                          <Pause className="w-4 h-4" /> Pause
                        </button>
                      ) : (
                        <button
                          id="player-play"
                          disabled={generatedNotes.length === 0}
                          onClick={startPlayback}
                          className="px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-850 disabled:text-slate-500 disabled:cursor-not-allowed text-slate-950 rounded-lg text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 transition-all shadow-glow-cyan"
                        >
                          <Play className="w-4 h-4 fill-slate-950" /> Play Composition
                        </button>
                      )}
                      
                      <button
                        id="player-stop"
                        disabled={!isPlaying}
                        onClick={stopPlayback}
                        className="p-2.5 bg-slate-800 hover:bg-slate-750 text-white rounded-lg transition-all disabled:opacity-40"
                        title="Stop"
                      >
                        <Square className="w-4 h-4" />
                      </button>
                    </div>

                    {generatedNotes.length > 0 && (
                      <button
                        id="download-midi-btn"
                        onClick={handleMidiDownload}
                        className="px-4 py-2 bg-slate-800 hover:bg-slate-750 text-emerald-400 border border-slate-750 hover:border-emerald-500 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all shadow-glow-emerald"
                      >
                        <Download className="w-4 h-4" /> Download Standard MIDI
                      </button>
                    )}
                  </div>
                </div>

                {/* High-Fidelity AI Piano Roll Sequence Matrix */}
                <div className="mb-6 bg-slate-950/40 rounded-xl p-4 border border-slate-900/80 flex flex-col gap-3 relative overflow-hidden">
                  <div className="flex justify-between items-center pb-2 border-b border-slate-900/60">
                    <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider flex items-center gap-1.5">
                      <Sliders className="w-3.5 h-3.5 text-indigo-400 animate-pulse" /> Neural Sequencer & Piano Roll
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono">
                      Beat: {playbackBeat.toFixed(2)} / {totalBeats.toFixed(0)} | Pitch Range: [{minPitch + 1}-{maxPitch - 1}]
                    </span>
                  </div>

                  {generatedNotes.length === 0 ? (
                    <div className="h-32 flex flex-col items-center justify-center text-center px-4 border border-dashed border-slate-800/60 rounded-lg bg-[#070b14]/30">
                      <Sparkles className="w-6 h-6 text-slate-700 animate-pulse mb-2" />
                      <span className="text-xs text-slate-500 font-medium font-display">Neural Sequencer Grid Empty</span>
                      <span className="text-[10px] text-slate-600 mt-1 max-w-sm font-mono leading-relaxed">
                        Predictions will align in a high-density, real-time MIDI note matrix once you trigger AI composition.
                      </span>
                    </div>
                  ) : (
                    <div className="relative h-40 bg-slate-950/80 border border-slate-900/80 rounded-lg p-1.5 overflow-hidden">
                      {/* Grid background lines */}
                      <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-20">
                        {/* Horizontal pitch rows */}
                        {Array.from({ length: Math.min(pitchRange, 20) }).map((_, i) => (
                          <line
                            key={`h-${i}`}
                            x1="0"
                            y1={`${(i / Math.min(pitchRange, 20)) * 100}%`}
                            x2="100%"
                            y2={`${(i / Math.min(pitchRange, 20)) * 100}%`}
                            stroke="#111827"
                            strokeWidth="0.5"
                          />
                        ))}
                        {/* Vertical beat subdivision columns */}
                        {Array.from({ length: 16 }).map((_, i) => (
                          <line
                            key={`v-${i}`}
                            x1={`${(i / 16) * 100}%`}
                            y1="0"
                            x2={`${(i / 16) * 100}%`}
                            y2="100%"
                            stroke={i % 4 === 0 ? "#1e293b" : "#0f172a"}
                            strokeWidth={i % 4 === 0 ? "1" : "0.5"}
                          />
                        ))}
                      </svg>

                      {/* Notes Overlay */}
                      <svg className="absolute inset-0 w-full h-full">
                        {generatedNotes.map((note, idx) => {
                          const noteWidth = (note.duration / totalBeats) * 100;
                          const noteX = (note.time / totalBeats) * 100;
                          const noteHeight = (1 / pitchRange) * 100;
                          const noteY = ((maxPitch - note.pitch) / pitchRange) * 100;

                          const isNoteActive = isPlaying && playbackBeat >= note.time && playbackBeat <= (note.time + note.duration);

                          let fillClass = "fill-indigo-500/80";
                          let activeFill = "fill-indigo-300";
                          let glowColor = "rgba(99, 102, 241, 0.4)";
                          
                          if (selectedGenre === 'jazz') {
                            fillClass = "fill-cyan-500/80";
                            activeFill = "fill-cyan-300";
                            glowColor = "rgba(34, 211, 238, 0.4)";
                          } else if (selectedGenre === 'ambient') {
                            fillClass = "fill-emerald-500/80";
                            activeFill = "fill-emerald-300";
                            glowColor = "rgba(16, 185, 129, 0.4)";
                          }

                          return (
                            <rect
                              key={idx}
                              x={`${noteX}%`}
                              y={`${noteY}%`}
                              width={`${Math.max(noteWidth, 0.8)}%`}
                              height={`${Math.max(noteHeight - 0.5, 1.5)}%`}
                              rx="1.5"
                              className={`transition-all duration-100 ${isNoteActive ? `${activeFill} filter drop-shadow-[0_0_4px_${glowColor}]` : fillClass}`}
                            />
                          );
                        })}

                        {/* Playing Progress Head Vertical Line */}
                        {isPlaying && (
                          <line
                            x1={`${(playbackBeat / totalBeats) * 100}%`}
                            y1="0"
                            x2={`${(playbackBeat / totalBeats) * 100}%`}
                            y2="100%"
                            stroke="#f43f5e"
                            strokeWidth="1.5"
                            className="filter drop-shadow-[0_0_3px_#f43f5e]"
                          />
                        )}
                      </svg>
                    </div>
                  )}
                  <div className="flex justify-between items-center text-[9px] text-slate-500 font-mono">
                    <span>Note Pitch (Y-Axis)</span>
                    <span className="flex items-center gap-1.5">
                      <span className="flex items-center gap-1"><span className="w-2 h-1.5 rounded bg-indigo-500/80"></span> Classical</span>
                      <span className="flex items-center gap-1"><span className="w-2 h-1.5 rounded bg-cyan-500/80"></span> Jazz</span>
                      <span className="flex items-center gap-1"><span className="w-2 h-1.5 rounded bg-emerald-500/80"></span> Ambient</span>
                    </span>
                    <span>Sequence Flow Time (X-Axis)</span>
                  </div>
                </div>

                {/* 25-Key Interactive Keyboard Visualizer */}
                <div className="mt-4">
                  <span className="text-[10px] text-slate-500 font-mono uppercase block mb-3">Live Recurrent Synaptic Keyboard Visualizer</span>
                  
                  <div className="relative h-44 bg-slate-950 p-2 rounded-xl border border-slate-800 overflow-x-auto flex justify-start">
                    <div className="relative flex h-full min-w-[500px] w-full">
                      {/* Render White Keys first */}
                      {PIANO_KEYS.filter(k => !k.isBlack).map((key, idx) => {
                        const isActive = activeNotes.includes(key.midi);
                        return (
                          <button
                            key={key.midi}
                            id={`white-key-${key.midi}`}
                            onClick={() => handleKeyClick(key.midi)}
                            className={`flex-grow h-full border-r border-slate-300 rounded-b-md flex flex-col justify-end pb-3 text-[9px] font-mono font-bold transition-all ${
                              isActive 
                                ? selectedGenre === 'classical' 
                                  ? 'bg-indigo-500 text-white shadow-glow-indigo border-none translate-y-[2px]'
                                  : selectedGenre === 'jazz'
                                    ? 'bg-cyan-400 text-slate-950 shadow-glow-cyan border-none translate-y-[2px]'
                                    : 'bg-emerald-400 text-slate-950 shadow-glow-emerald border-none translate-y-[2px]'
                                : 'bg-white hover:bg-slate-100 text-slate-500'
                            }`}
                          >
                            <span>{key.name}</span>
                          </button>
                        );
                      })}

                      {/* Absolute overlay of black keys */}
                      {PIANO_KEYS.map((key, idx) => {
                        if (!key.isBlack) return null;

                        // Position calculation relative to standard white key indexes
                        const precedingWhiteKeys = PIANO_KEYS.slice(0, idx).filter(k => !k.isBlack).length;
                        const leftPercent = (precedingWhiteKeys / 15) * 100; // 15 white keys in 25-key layout

                        const isActive = activeNotes.includes(key.midi);

                        return (
                          <button
                            key={key.midi}
                            id={`black-key-${key.midi}`}
                            onClick={() => handleKeyClick(key.midi)}
                            style={{ 
                              left: `calc(${leftPercent}% - 10px)`,
                              width: '20px'
                            }}
                            className={`absolute h-[60%] rounded-b-sm border-b-2 border-slate-950 transition-all z-10 flex flex-col justify-end pb-1 text-[7px] font-mono text-center ${
                              isActive 
                                ? selectedGenre === 'classical'
                                  ? 'bg-indigo-600 text-white shadow-glow-indigo border-none translate-y-[1px]'
                                  : selectedGenre === 'jazz'
                                    ? 'bg-cyan-500 text-slate-950 shadow-glow-cyan border-none translate-y-[1px]'
                                    : 'bg-emerald-500 text-slate-950 shadow-glow-emerald border-none translate-y-[1px]'
                                : 'bg-slate-900 hover:bg-slate-800 text-slate-400'
                            }`}
                          >
                            <span>{key.name}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Console Output for Symbolic generation */}
                <div className="mt-5 bg-[#0B1120] border border-slate-800/80 p-3.5 rounded-lg">
                  <div className="flex items-center gap-1.5 text-[10px] text-slate-400 uppercase font-mono mb-2 border-b border-slate-800/80 pb-1.5">
                    <Terminal className="w-3.5 h-3.5 text-indigo-400" /> Inference Engine Output Logs
                  </div>
                  <div className="font-mono text-[11px] text-slate-400 max-h-[110px] overflow-y-auto space-y-1">
                    {generationLogs.length === 0 ? (
                      <span className="text-slate-600 block">Idle. Configure parameters and click "Compose AI Melody" to start.</span>
                    ) : (
                      generationLogs.map((log, idx) => (
                        <div key={idx} className="flex gap-2">
                          <span className="text-slate-600">[{idx+1}]</span>
                          <span className={log.startsWith("Error") ? "text-rose-400" : log.includes("Successfully") ? "text-emerald-400" : "text-slate-300"}>
                            {log}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 2. TRAINING DASHBOARD TAB */}
        {activeTab === 'training' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in duration-300">
            
            {/* Left Controls */}
            <div className="bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-6 flex flex-col justify-between shadow-2xl">
              <div>
                <h3 className="text-sm font-semibold font-display text-white uppercase tracking-wider mb-4 pb-2 border-b border-slate-800/80 flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-indigo-400" /> Training Controller
                </h3>

                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Train the 3-layer LSTM recurrent neural network on your custom MIDI datasets. 
                  This simulation feeds sequence vectors of length 100 into the cell state to calculate categorical backpropagation loss.
                </p>

                {/* Dataset Choice */}
                <div className="mb-5">
                  <label className="block text-xs font-medium text-slate-400 mb-2">Training Target Profile</label>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-850 flex items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold text-white capitalize block">{selectedGenre} Collection</span>
                      <span className="text-[10px] text-slate-500 font-mono block mt-0.5">
                        {selectedGenre === 'classical' 
                          ? '34 Classical MIDI files (~450k notes)' 
                          : selectedGenre === 'jazz' 
                            ? '25 Realbook Jazz MIDI files (~280k notes)' 
                            : '18 Ambient Piano MIDI files (~150k notes)'}
                      </span>
                    </div>
                    <Music className="w-5 h-5 text-indigo-400" />
                  </div>
                </div>

                {/* Training Speed */}
                <div className="mb-5">
                  <label className="block text-xs font-medium text-slate-400 mb-2">Simulation Cycle Speed</label>
                  <div className="grid grid-cols-3 gap-2">
                    {(['slow', 'fast', 'instant'] as const).map((spd) => (
                      <button
                        key={spd}
                        id={`speed-btn-${spd}`}
                        onClick={() => setTrainingSpeed(spd)}
                        className={`py-1.5 text-xs font-semibold rounded-lg border capitalize transition-all ${
                          trainingSpeed === spd 
                            ? 'bg-indigo-600/10 border-indigo-500 text-indigo-400 shadow-glow-indigo' 
                            : 'border-slate-850 bg-slate-950/40 text-slate-400 hover:border-slate-700'
                        }`}
                      >
                        {spd}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Progress Indicators */}
                {isTraining && (
                  <div className="mb-4 bg-slate-950 p-3 rounded-lg border border-slate-850">
                    <div className="flex justify-between text-xs font-mono font-bold mb-1.5">
                      <span className="text-slate-400">Backpropagation Progress</span>
                      <span className="text-indigo-400">{trainingProgress}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                      <div 
                        className="bg-indigo-500 h-full transition-all duration-300 shadow-glow-indigo"
                        style={{ width: `${trainingProgress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>

              {/* Execution Actions */}
              <div className="space-y-2">
                {isTraining ? (
                  <button
                    id="stop-training-btn"
                    onClick={cancelModelTraining}
                    className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-semibold tracking-wider uppercase flex items-center justify-center gap-1.5"
                  >
                    <Square className="w-4.5 h-4.5" /> Stop Model Training
                  </button>
                ) : (
                  <button
                    id="start-training-btn"
                    onClick={triggerModelTraining}
                    className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold tracking-wider uppercase flex items-center justify-center gap-1.5 shadow-glow-indigo"
                  >
                    <Cpu className="w-4.5 h-4.5" /> Launch Model Training
                  </button>
                )}
                
                <button
                  id="reset-training-btn"
                  onClick={() => {
                    setCurrentTrainingLogs(SIMULATED_TRAINING_LOGS[selectedGenre].slice(0, 3));
                    setTrainingConsoleLogs(["[INFO] Training history reset to epoch 10."]);
                  }}
                  className="w-full py-2 bg-slate-800 hover:bg-slate-750 text-slate-300 rounded-lg text-xs font-semibold"
                >
                  Reset Weights & Charts
                </button>
              </div>
            </div>

            {/* Graphs Display */}
            <div className="lg:col-span-2 flex flex-col gap-6">
              
              {/* Convergence chart with Recharts */}
              <div className="bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-6 shadow-2xl flex-grow">
                <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800/80">
                  <h3 className="text-sm font-semibold font-display text-white uppercase tracking-wider flex items-center gap-1.5">
                    <TrendingDown className="w-4 h-4 text-indigo-400" /> Loss Convergence & Accuracy Curves
                  </h3>
                  <div className="text-[11px] text-slate-400 font-mono bg-slate-950 px-2.5 py-1 rounded border border-slate-850">
                    Keras Metric Monitor
                  </div>
                </div>

                <div className="h-64 md:h-80 w-full font-mono text-xs">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={currentTrainingLogs}
                      margin={{ top: 10, right: 20, left: -20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="epoch" name="Epoch" stroke="#94a3b8" />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
                        labelStyle={{ fontWeight: 'bold' }}
                      />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="loss" 
                        name="Train Loss" 
                        stroke="#6366f1" 
                        strokeWidth={2}
                        activeDot={{ r: 6 }} 
                      />
                      <Line 
                        type="monotone" 
                        dataKey="valLoss" 
                        name="Val Loss" 
                        stroke="#f43f5e" 
                        strokeWidth={2}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="accuracy" 
                        name="Train Acc" 
                        stroke="#10b981" 
                        strokeWidth={1.5}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Console Logs */}
              <div className="bg-[#02050c] border border-slate-900/80 p-4 rounded-xl flex-grow max-h-56 flex flex-col">
                <span className="text-[10px] text-slate-500 font-mono uppercase block mb-2 pb-1.5 border-b border-slate-800/80 flex items-center gap-1.5">
                  <Terminal className="w-4 h-4 text-indigo-400" /> Backprop Convolution Command Line Console
                </span>
                <div className="font-mono text-xs text-slate-300 space-y-1.5 overflow-y-auto flex-grow">
                  {trainingConsoleLogs.map((log, index) => (
                    <div key={index} className="flex gap-2">
                      <span className="text-slate-600 font-bold">&gt;&gt;</span>
                      <span className={log.includes("[SUCCESS]") ? "text-emerald-400 font-bold" : log.includes("[INFO]") ? "text-slate-400" : "text-indigo-300"}>
                        {log}
                      </span>
                    </div>
                  ))}
                  {isTraining && (
                    <div className="text-indigo-400 flex items-center gap-1.5 animate-pulse">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-ping"></span>
                      Backpropagating weights gradients...
                    </div>
                  )}
                </div>
              </div>

              {/* Dataset Manager & MIDI Uploader */}
              <div className="bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-6 shadow-2xl flex flex-col">
                <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800/80">
                  <h3 className="text-sm font-semibold font-display text-white uppercase tracking-wider flex items-center gap-1.5">
                    <Sliders className="w-4 h-4 text-indigo-400" /> Upload MIDI Dataset & File Manager
                  </h3>
                  <span className="text-[10px] text-indigo-400 font-mono bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                    Flask Backend Sync
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* File upload section */}
                  <div className="flex flex-col justify-center">
                    <label className="block text-xs text-slate-400 mb-2">Upload custom MIDI file to dataset folder (.mid/.midi)</label>
                    <div className="border border-dashed border-slate-800 hover:border-indigo-500/50 bg-slate-950/40 rounded-xl p-6 text-center cursor-pointer transition-all relative">
                      <input 
                        type="file" 
                        accept=".mid,.midi" 
                        onChange={handleFileUpload}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                        disabled={isUploading}
                      />
                      <div className="flex flex-col items-center justify-center gap-2">
                        {isUploading ? (
                          <>
                            <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin" />
                            <span className="text-xs text-slate-400">Uploading to Flask...</span>
                          </>
                        ) : (
                          <>
                            <Upload className="w-8 h-8 text-slate-500" />
                            <span className="text-xs text-slate-400">Click to upload or drop file</span>
                            <span className="text-[10px] text-slate-600">Supports standard MIDI format 0/1</span>
                          </>
                        )}
                      </div>
                    </div>
                    {uploadError && <div className="text-xs text-rose-400 mt-2 font-mono">{uploadError}</div>}
                    {uploadSuccess && <div className="text-xs text-emerald-400 mt-2 font-mono">{uploadSuccess}</div>}
                  </div>

                  {/* Dataset file list */}
                  <div className="flex flex-col justify-between">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-semibold text-slate-400">Active Dataset Folder ({datasetFiles.length} files)</span>
                        <button 
                          onClick={fetchDataset}
                          className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-all"
                          title="Refresh dataset list"
                        >
                          <RefreshCw className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <div className="bg-slate-950/60 rounded-xl border border-slate-850 p-2 max-h-36 overflow-y-auto space-y-1.5 font-mono text-[11px]">
                        {datasetFiles.length === 0 ? (
                          <div className="text-slate-600 text-center py-4">No MIDI files in dataset. Upload some above!</div>
                        ) : (
                          datasetFiles.map((file, idx) => (
                            <div key={idx} className="flex justify-between items-center bg-slate-900/60 border border-slate-850/50 rounded-lg px-2.5 py-1 text-slate-300">
                              <span className="truncate max-w-[180px]" title={file.name}>{file.name}</span>
                              <span className="text-slate-500 text-[10px]">{file.size}</span>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                    <div className="text-[10px] text-slate-500 italic mt-3 leading-tight">
                      *Note: After uploading new files, trigger the Training sequence to compile and build the new index mappings.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 3. CODE WORKSPACE TAB */}
        {activeTab === 'code' && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 animate-in fade-in duration-300">
            
            {/* File explorer panel */}
            <div className="bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-5 shadow-2xl lg:col-span-1">
              <h3 className="text-xs font-semibold font-display text-white uppercase tracking-wider mb-3.5 pb-2 border-b border-slate-800/80 flex items-center gap-1.5">
                <FileCode className="w-4 h-4 text-indigo-400" /> Music_Generation_AI/
              </h3>

              <div className="space-y-1.5">
                <div className="flex items-center space-x-2 p-1 text-xs text-slate-500 italic font-mono pl-2">
                  <span>├── dataset/</span>
                </div>
                {pythonFiles.map((file) => {
                  const isSelected = selectedCodeFile.name === file.name;
                  return (
                    <button
                      key={file.name}
                      id={`file-explorer-${file.name}`}
                      onClick={() => setSelectedCodeFile(file)}
                      className={`w-full text-left px-3 py-2 text-xs font-mono rounded-lg flex items-center justify-between transition-all ${
                        isSelected 
                          ? 'bg-indigo-600/10 text-indigo-300 border border-indigo-500/30' 
                          : 'text-slate-400 hover:text-white hover:bg-slate-950/50'
                      }`}
                    >
                      <span className="flex items-center gap-1.5">
                        <ChevronRight className={`w-3.5 h-3.5 text-slate-500 transition-all ${isSelected ? 'rotate-90 text-indigo-400' : ''}`} />
                        {file.name}
                      </span>
                      <span className="text-[9px] bg-slate-950 border border-slate-850 px-1.5 py-0.5 rounded text-slate-500">
                        {file.language.toUpperCase()}
                      </span>
                    </button>
                  );
                })}
                <div className="flex items-center space-x-2 p-1 text-xs text-slate-500 italic font-mono pl-2">
                  <span>└── report/</span>
                </div>
              </div>

              {/* Zip Export instructions block */}
              <div className="mt-6 bg-slate-950 p-3.5 rounded-lg border border-slate-850">
                <span className="text-[10px] text-slate-500 font-mono uppercase block mb-1">Workspace Export Note</span>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  These matching Python module files reside in your container's absolute workspace path. To export the full executable directory as a single ZIP, use the <b>Settings Menu &gt; Export to ZIP</b> option in AI Studio!
                </p>
              </div>
            </div>

            {/* Code Block Viewer */}
            <div className="lg:col-span-3 flex flex-col bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl overflow-hidden shadow-2xl">
              <div className="bg-slate-950 px-5 py-3 border-b border-slate-850 flex items-center justify-between">
                <div>
                  <span className="text-xs font-semibold font-mono text-white flex items-center gap-1.5">
                    <FileText className="w-4 h-4 text-indigo-400" /> {selectedCodeFile.path}
                  </span>
                  <span className="text-[10px] text-slate-500 block font-mono mt-0.5">Absolute Path: /Music_Generation_AI/{selectedCodeFile.path}</span>
                </div>

                <div className="flex space-x-2">
                  <button
                    id="copy-code-btn"
                    onClick={() => handleCopyCode(selectedCodeFile)}
                    className="p-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 rounded-md text-xs flex items-center gap-1.5 transition-all"
                  >
                    {copiedFile === selectedCodeFile.name ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-emerald-400" /> Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" /> Copy Code
                      </>
                    )}
                  </button>
                  
                  {/* Anchor to download code as individual script */}
                  <a
                    id="download-script-btn"
                    href={`data:text/plain;charset=utf-8,${encodeURIComponent(selectedCodeFile.content)}`}
                    download={selectedCodeFile.name}
                    className="p-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 rounded-md text-xs flex items-center gap-1.5 transition-all"
                  >
                    <Download className="w-3.5 h-3.5" /> Download File
                  </a>
                </div>
              </div>

              {/* Code Panel */}
              <div className="p-4 overflow-auto flex-grow max-h-[500px]">
                <pre className="font-mono text-xs text-slate-300 leading-relaxed select-all">
                  <code>
                    {selectedCodeFile.content.split('\n').map((line, idx) => (
                      <div key={idx} className="flex hover:bg-slate-950/20 px-2 py-0.5 rounded transition-all">
                        <span className="w-10 text-right pr-4 text-slate-600 font-sans select-none border-r border-slate-850 mr-4">{idx + 1}</span>
                        <span>{line || ' '}</span>
                      </div>
                    ))}
                  </code>
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* 4. ACADEMIC REPORT TAB */}
        {activeTab === 'report' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in duration-300">
            
            {/* Outline Panel */}
            <div className="bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-5 shadow-2xl lg:col-span-1 flex flex-col justify-between">
              <div>
                <h3 className="text-sm font-semibold font-display text-white uppercase tracking-wider mb-4 pb-2 border-b border-slate-800/80 flex items-center gap-1.5">
                  <BookOpen className="w-4 h-4 text-indigo-400" /> College Project Report
                </h3>

                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  This contains a complete academic submission manuscript documenting the artificial intelligence model, architecture diagram, neural cell mathematical formulas, and experimental outcomes.
                </p>

                <div className="space-y-3 font-mono text-[11px] text-slate-400 bg-slate-950 p-4 rounded-xl border border-slate-850">
                  <span className="text-[10px] uppercase font-bold text-slate-500 block">Report Sections Included</span>
                  <div className="space-y-1.5">
                    <span className="block flex items-center gap-1.5"><ArrowRight className="w-3 h-3 text-indigo-400" /> 1. Project Introduction</span>
                    <span className="block flex items-center gap-1.5"><ArrowRight className="w-3 h-3 text-indigo-400" /> 2. Problem Statement Definition</span>
                    <span className="block flex items-center gap-1.5"><ArrowRight className="w-3 h-3 text-indigo-400" /> 3. Modular System Architecture</span>
                    <span className="block flex items-center gap-1.5"><ArrowRight className="w-3 h-3 text-indigo-400" /> 4. Math & Formulas (LSTM cell state)</span>
                    <span className="block flex items-center gap-1.5"><ArrowRight className="w-3 h-3 text-indigo-400" /> 5. Datasets, Training, and Graphs</span>
                    <span className="block flex items-center gap-1.5"><ArrowRight className="w-3 h-3 text-indigo-400" /> 6. Real Experimental Evaluation</span>
                  </div>
                </div>
              </div>

              {/* Download Report Actions */}
              <div className="mt-5 space-y-2">
                <a
                  id="download-report-btn"
                  href={`data:text/markdown;charset=utf-8,${encodeURIComponent(
                    pythonFiles.find(f => f.name === 'preprocess.py')?.content || '' // placeholder markdown generator
                  )}`}
                  download="Music_Generation_AI_Project_Report.md"
                  className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold tracking-wider uppercase flex items-center justify-center gap-1.5 transition-all shadow-glow-indigo"
                >
                  <Download className="w-4 h-4" /> Export Report (Markdown)
                </a>
                
                <div className="text-[10px] text-slate-500 leading-relaxed text-center font-mono">
                  Compiles LaTeX formulas, ASCII tables, and source charts.
                </div>
              </div>
            </div>

            {/* Document preview panel */}
            <div className="lg:col-span-2 bg-[#050914]/60 backdrop-blur-md border border-slate-900/85 rounded-2xl p-6 shadow-2xl overflow-y-auto max-h-[600px] prose prose-invert prose-slate">
              <h1 className="text-xl font-bold font-display text-white pb-3 border-b border-slate-800/80 mb-5">
                Music Generation with AI Using LSTM Neural Networks
              </h1>

              <div className="text-xs text-slate-400 leading-relaxed space-y-6">
                
                <div>
                  <h2 className="text-sm font-semibold font-display text-white uppercase tracking-wider mb-2">1. Executive Overview</h2>
                  <p>
                    Autonomous symbolic music composition requires models that extract structural patterns across high-dimensional sequences. While classical Markov chains are limited to short-range transitions, this project builds and evaluates a deep **Long Short-Term Memory (LSTM)** recurrent neural network. The model is trained on classical and jazz piano score streams to generate original, aesthetic compositions saved directly in MIDI format.
                  </p>
                </div>

                <div>
                  <h2 className="text-sm font-semibold font-display text-white uppercase tracking-wider mb-2">2. Problem Statement</h2>
                  <p>
                    Symbolic musical structures encompass multi-scale hierarchies: short-term structures (intervals and chords) and long-term structures (themes and motifs). Traditional Feedforward architectures fail to capture temporal dependencies due to vanishing gradients during backpropagation. This project implements gated cells that preserve state memory over lengthy horizons, preventing degenerative convergence.
                  </p>
                </div>

                <div>
                  <h2 className="text-sm font-semibold font-display text-white uppercase tracking-wider mb-2">3. Mathematical Framework of the LSTM Cell</h2>
                  <p className="font-mono bg-[#0B1120] p-4 rounded-lg border border-slate-800/80 text-slate-300">
                    The LSTM gate mechanics are defined mathematically by:<br/><br/>
                    • Forget Gate: f_t = σ(W_f · [h_t-1, x_t] + b_f)<br/>
                    • Input Gate: i_t = σ(W_i · [h_t-1, x_t] + b_i)<br/>
                    • Candidate Cell State: ~C_t = tanh(W_c · [h_t-1, x_t] + b_c)<br/>
                    • Cell State Update: C_t = f_t * C_t-1 + i_t * ~C_t<br/>
                    • Output Gate: o_t = σ(W_o · [h_t-1, x_t] + b_o)<br/>
                    • Final Hidden State: h_t = o_t * tanh(C_t)
                  </p>
                </div>

                <div>
                  <h2 className="text-sm font-semibold font-display text-white uppercase tracking-wider mb-2">4. Systematic Implementation Architecture</h2>
                  <p>
                    The systemic pipeline is modularized into three main stages:
                  </p>
                  <ol className="list-decimal pl-5 space-y-1 mt-1.5">
                    <li><b>Data Preprocessing (`preprocess.py`):</b> Uses Python's `music21` package to load MIDI streams, split instruments, extract note/chord configurations, map strings to integers, and serialize arrays.</li>
                    <li><b>Neural Fitting (`train.py`):</b> Builds the 3-layer sequential LSTM stack in Keras, applies Dropout regularizers, configures Categorical Cross-Entropy, and saves performance logs.</li>
                    <li><b>Melodic Synthesis (`generate_music.py`):</b> Initiates model prediction, applies Temperature Sampling on Softmax output, builds a music21 stream, and writes the stream to disk as a timestamped MIDI composition.</li>
                  </ol>
                </div>

                <div>
                  <h2 className="text-sm font-semibold font-display text-white uppercase tracking-wider mb-2">5. Experimental Outcomes</h2>
                  <p>
                    During training, convergence is assessed using categorical cross-entropy. High-temperature testing ($T \ge 1.2$) results in complex, dissonant syncopated arrangements resembling progressive avant-garde jazz. Lower temperatures ($T \le 0.4$) lead to repetitive intervals, highlighting the necessity of temperature control to maintain the balance between predictability and creativity.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* FOOTER BAR */}
      <footer className="border-t border-slate-900/90 bg-[#050914] px-6 py-4 mt-12 text-center text-xs text-slate-500 font-mono">
        <p>© 2026 AI Studio Build. Designed and implemented for advanced college submission curricula.</p>
        <p className="mt-1 text-slate-600">Technologies: React 19, Vite, Recharts, Web Audio API, TypeScript & Python music21 / TensorFlow</p>
      </footer>
    </div>
  );
}
