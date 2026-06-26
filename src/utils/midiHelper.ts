import { NoteEvent } from '../types';

/**
 * Encodes a numeric value as a MIDI Variable Length Quantity (VLQ).
 */
function encodeVLQ(value: number): number[] {
  const bytes: number[] = [];
  bytes.push(value & 0x7f);
  while ((value >>= 7) > 0) {
    bytes.push((value & 0x7f) | 0x80);
  }
  return bytes.reverse();
}

/**
 * Compiles an array of NoteEvents into a valid binary Standard MIDI File (Format 0).
 */
export function compileMIDI(notes: NoteEvent[]): Uint8Array {
  const headerChunk = [
    0x4d, 0x54, 0x68, 0x64, // "MThd" header
    0x00, 0x00, 0x00, 0x06, // Chunk length (6 bytes)
    0x00, 0x00,             // Format type (0 = single track)
    0x00, 0x01,             // Number of tracks (1)
    0x01, 0xe0              // Time division (480 ticks per quarter note)
  ];

  const trackEvents: number[] = [];
  
  // Set tempo meta event: FF 51 03 tttttt (microseconds per quarter note)
  // Let's set tempo to 120 BPM = 500,000 microseconds per quarter note (0x07a120)
  trackEvents.push(0x00); // delta-time = 0
  trackEvents.push(0xff, 0x51, 0x03, 0x07, 0xa1, 0x20);

  // Keep track of active note turn-offs
  // Since we want standard step sequencing, we can sort absolute timings.
  interface MIDIAction {
    absoluteTick: number;
    type: 'on' | 'off';
    pitch: number;
    velocity: number;
  }

  const actions: MIDIAction[] = [];
  const ticksPerBeat = 480;

  notes.forEach(note => {
    const startTick = Math.round(note.time * ticksPerBeat);
    const endTick = Math.round((note.time + note.duration) * ticksPerBeat);
    
    actions.push({
      absoluteTick: startTick,
      type: 'on',
      pitch: note.pitch,
      velocity: note.velocity
    });
    
    actions.push({
      absoluteTick: endTick,
      type: 'off',
      pitch: note.pitch,
      velocity: 0
    });
  });

  // Sort actions by absolute time.
  // If absolute times are equal, turn off actions should occur before turn on actions to avoid cutoff overlap.
  actions.sort((a, b) => {
    if (a.absoluteTick === b.absoluteTick) {
      return a.type === 'off' ? -1 : 1;
    }
    return a.absoluteTick - b.absoluteTick;
  });

  let previousTick = 0;

  actions.forEach(act => {
    const deltaTick = act.absoluteTick - previousTick;
    previousTick = act.absoluteTick;

    // Write delta-time in VLQ
    const vlqBytes = encodeVLQ(deltaTick);
    trackEvents.push(...vlqBytes);

    if (act.type === 'on') {
      // Note On: 0x90 (Channel 1, channel indices are 0-indexed in status bytes: 0x90 is Channel 1)
      trackEvents.push(0x90, act.pitch, act.velocity);
    } else {
      // Note Off: 0x80 (Channel 1)
      trackEvents.push(0x80, act.pitch, 0x40);
    }
  });

  // End of track meta-event: delta = 0, FF 2F 00
  trackEvents.push(0x00);
  trackEvents.push(0xff, 0x2f, 0x00);

  const trackLength = trackEvents.length;
  const trackChunkHeader = [
    0x4d, 0x54, 0x72, 0x6b, // "MTrk" header
    (trackLength >> 24) & 0xff,
    (trackLength >> 16) & 0xff,
    (trackLength >> 8) & 0xff,
    trackLength & 0xff
  ];

  const fullMidi = new Uint8Array([...headerChunk, ...trackChunkHeader, ...trackEvents]);
  return fullMidi;
}

/**
 * Triggers a browser download of compiled MIDI bytes.
 */
export function downloadMIDIFile(notes: NoteEvent[], filename: string): void {
  const bytes = compileMIDI(notes);
  const blob = new Blob([bytes], { type: 'audio/midi' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  
  // Clean up
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
