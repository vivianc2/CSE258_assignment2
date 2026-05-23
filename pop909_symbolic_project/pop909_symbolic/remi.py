"""REMI-style tokenizer for melody-only sequences.

REMI (Huang & Yang, "Pop Music Transformer", 2020) replaces fixed-grid frame
tokens with event tokens: Bar, Position, Pitch, Duration. This preserves note
onsets and durations explicitly, which is the main reason REMI-based language
models sound much more musical than frame-based LSTMs.

Vocabulary layout:

  PAD=0  BOS=1  EOS=2  BAR=3
  POS_0..POS_{STEPS_PER_BAR-1}                           -> [4, 4+STEPS_PER_BAR)
  PITCH_{PITCH_MIN}..PITCH_{PITCH_MAX}                   -> next 88 ids
  DUR_1..DUR_{MAX_DUR}                                    -> next MAX_DUR ids

We assume 4/4 time and a 16th-note grid (STEPS_PER_BAR=16).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import bisect, math

from .midi_io import Note

# --- vocabulary -------------------------------------------------------------

STEPS_PER_BAR = 16          # 4 beats x 4 sixteenths
MAX_DUR = 32                # up to 2 bars
PITCH_MIN, PITCH_MAX = 21, 108  # 88 piano keys

PAD_ID, BOS_ID, EOS_ID, BAR_ID = 0, 1, 2, 3
POS_BASE = 4
PITCH_BASE = POS_BASE + STEPS_PER_BAR        # 20
DUR_BASE = PITCH_BASE + (PITCH_MAX - PITCH_MIN + 1)  # 108
VOCAB_SIZE = DUR_BASE + MAX_DUR              # 140

def pos_id(p: int) -> int: return POS_BASE + max(0, min(STEPS_PER_BAR - 1, int(p)))
def pitch_id(m: int) -> int: return PITCH_BASE + max(0, min(PITCH_MAX - PITCH_MIN, int(m) - PITCH_MIN))
def dur_id(d: int) -> int: return DUR_BASE + max(0, min(MAX_DUR - 1, int(d) - 1))

def token_kind(tok: int) -> str:
    if tok < POS_BASE: return ["PAD","BOS","EOS","BAR"][tok]
    if tok < PITCH_BASE: return "POS"
    if tok < DUR_BASE: return "PITCH"
    if tok < VOCAB_SIZE: return "DUR"
    return "UNK"

# --- encoding ---------------------------------------------------------------

@dataclass
class QuantizedNote:
    step: int          # absolute 16th-note step since song start
    pitch: int         # MIDI pitch in [PITCH_MIN, PITCH_MAX]
    duration: int      # in 16th notes, >=1

def _step_from_beat(beat: float, beat_times: Sequence[float]) -> float:
    """Map a beat-time (in beats since 0) to fractional 16th-note steps.

    POP909 beat_times are the actual beat onsets, so 1 beat = 4 sixteenths even
    if tempo varies. We use the local beat duration around the time of the note.
    """
    return beat * 4.0  # frame_melody already feeds us beat-aligned timings

def quantize_melody(notes: Sequence[Note], beat_times: Sequence[float]) -> List[QuantizedNote]:
    """Quantize notes to a 16th-note grid using the song's beat times.

    Each note's start in seconds is converted to a fractional beat index by
    linear interpolation between beat onsets, then multiplied by 4 to get 16th
    steps. Notes are clipped to the supported pitch range.
    """
    if not notes:
        return []
    if not beat_times or len(beat_times) < 2:
        return []
    bt = list(beat_times)
    def to_step(t: float) -> float:
        i = bisect.bisect_right(bt, t) - 1
        i = max(0, min(len(bt) - 2, i))
        span = bt[i + 1] - bt[i]
        if span <= 0:
            return i * 4.0
        frac = (t - bt[i]) / span
        return (i + frac) * 4.0
    out: List[QuantizedNote] = []
    for n in notes:
        if not (PITCH_MIN <= n.pitch <= PITCH_MAX):
            continue
        s = int(round(to_step(n.start)))
        e = int(round(to_step(n.end)))
        d = max(1, min(MAX_DUR, e - s))
        out.append(QuantizedNote(step=max(0, s), pitch=int(n.pitch), duration=d))
    out.sort(key=lambda q: (q.step, -q.pitch))
    # If two notes share the same step, keep the higher pitch (melody = top voice).
    dedup: List[QuantizedNote] = []
    for q in out:
        if dedup and dedup[-1].step == q.step:
            continue
        dedup.append(q)
    return dedup

def encode_remi(notes: Sequence[Note], beat_times: Sequence[float], add_bos_eos: bool = True) -> List[int]:
    """Encode a melody (list of Notes + beat timing) into a REMI token sequence."""
    qns = quantize_melody(notes, beat_times)
    tokens: List[int] = []
    if add_bos_eos:
        tokens.append(BOS_ID)
    last_bar = -1
    for q in qns:
        bar_idx = q.step // STEPS_PER_BAR
        pos_in_bar = q.step % STEPS_PER_BAR
        while last_bar < bar_idx:
            tokens.append(BAR_ID)
            last_bar += 1
        tokens.append(pos_id(pos_in_bar))
        tokens.append(pitch_id(q.pitch))
        tokens.append(dur_id(q.duration))
    if add_bos_eos:
        tokens.append(EOS_ID)
    return tokens

# --- decoding ---------------------------------------------------------------

@dataclass
class DecodedNote:
    start_step: int
    pitch: int
    duration: int

def decode_remi(tokens: Sequence[int]) -> List[DecodedNote]:
    """Decode tokens back into notes (with absolute 16th-step start times).

    The decoder is lenient: any unexpected token order is silently skipped, so
    sampler outputs do not crash even when the model emits malformed runs.
    """
    notes: List[DecodedNote] = []
    bar_idx = 0
    cur_pos: Optional[int] = None
    cur_pitch: Optional[int] = None
    started_bars = False
    for t in tokens:
        if t == BAR_ID:
            if started_bars:
                bar_idx += 1
            else:
                started_bars = True
                bar_idx = 0
        elif POS_BASE <= t < PITCH_BASE:
            cur_pos = t - POS_BASE
            cur_pitch = None
        elif PITCH_BASE <= t < DUR_BASE:
            cur_pitch = (t - PITCH_BASE) + PITCH_MIN
        elif DUR_BASE <= t < VOCAB_SIZE:
            if cur_pos is None or cur_pitch is None:
                continue
            d = (t - DUR_BASE) + 1
            start_step = bar_idx * STEPS_PER_BAR + cur_pos
            notes.append(DecodedNote(start_step=start_step, pitch=int(cur_pitch), duration=int(d)))
            cur_pitch = None  # require a new Pitch before another Duration
        # PAD / BOS / EOS / anything else: ignore
    return notes

def decoded_notes_to_midi_notes(decoded: Sequence[DecodedNote], tempo_bpm: float = 105.0) -> List[Note]:
    """Convert decoded REMI notes into `Note` objects in beat-time (start/end in beats)."""
    # 16 steps per bar = 4 beats; so 1 step = 0.25 beats.
    out: List[Note] = []
    for d in decoded:
        start_beats = d.start_step * 0.25
        end_beats = (d.start_step + d.duration) * 0.25
        out.append(Note(pitch=int(d.pitch), start=start_beats, end=end_beats, velocity=88))
    out.sort(key=lambda n: (n.start, -n.pitch))
    return out
