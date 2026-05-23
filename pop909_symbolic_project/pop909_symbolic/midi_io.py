from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import math
import mido
from .constants import REST, NOTE_TO_PC, QUALITY_INTERVALS

@dataclass
class Note:
    pitch: int
    start: float
    end: float
    velocity: int = 80

@dataclass
class SongData:
    song_id: str
    midi_path: Path
    melody_notes: List[Note]
    beat_times: List[float]
    downbeats: List[int]
    chord_segments: List[Tuple[float, float, str]]
    key: Optional[Tuple[str, str]] = None
    transpose_delta: int = 0

def list_song_dirs(pop909_root: str | Path) -> List[Path]:
    root = Path(pop909_root)
    if not root.exists():
        raise FileNotFoundError(f"POP909 root does not exist: {root}")
    return [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.isdigit() and (p / f"{p.name}.mid").exists()]

def _track_name(track: mido.MidiTrack) -> str:
    for msg in track:
        if msg.type == "track_name":
            return str(msg.name).strip().upper()
    return ""

def extract_notes_by_track(midi_path: str | Path) -> Dict[str, List[Note]]:
    midi_path = Path(midi_path)
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    result: Dict[str, List[Note]] = {}
    for idx, track in enumerate(mid.tracks):
        name = _track_name(track) or f"TRACK_{idx}"
        abs_ticks = 0
        active: Dict[Tuple[int, int], Tuple[int, int]] = {}
        notes: List[Note] = []
        for msg in track:
            abs_ticks += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(getattr(msg, "channel", 0), msg.note)] = (abs_ticks, msg.velocity)
            elif msg.type in ("note_off", "note_on") and hasattr(msg, "note"):
                key = (getattr(msg, "channel", 0), msg.note)
                if key in active:
                    start_ticks, vel = active.pop(key)
                    if abs_ticks > start_ticks:
                        notes.append(Note(int(msg.note), start_ticks/tpb, abs_ticks/tpb, int(vel)))
        result[name] = sorted(notes, key=lambda n: (n.start, n.pitch))
    return result

def choose_melody_track(notes_by_track: Dict[str, List[Note]]) -> List[Note]:
    for key, notes in notes_by_track.items():
        if "MELODY" in key:
            return notes
    nonempty = [(k, v) for k, v in notes_by_track.items() if v]
    if not nonempty:
        return []
    _, notes = max(nonempty, key=lambda kv: sum(n.pitch for n in kv[1]) / max(1, len(kv[1])))
    return notes

def parse_beat_file(path: str | Path) -> Tuple[List[float], List[int]]:
    path = Path(path)
    if not path.exists():
        return [], []
    vals = path.read_text(encoding="utf-8").strip().split()
    times, downs = [], []
    for i in range(0, len(vals)-2, 3):
        try:
            times.append(float(vals[i])); downs.append(int(round(float(vals[i+2]))))
        except ValueError:
            pass
    return times, downs

_PC_TO_NOTE = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]

def parse_key_file(path: str | Path) -> Optional[Tuple[str, str]]:
    """Return the dominant (root_name, mode) from POP909 `key_audio.txt`.

    The file has lines of `start end label` where label is like "Gb:maj" or "A:min".
    Some songs have multiple segments; we pick the (root, mode) with the most total
    duration so that local modulations don't dominate normalization.
    """
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip().splitlines()
    durations: Dict[Tuple[str, str], float] = {}
    for line in text:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        label = parts[2]
        if ":" in label:
            root, mode = label.split(":", 1)
        else:
            root, mode = label, "maj"
        durations[(root, mode)] = durations.get((root, mode), 0.0) + max(0.0, b - a)
    if not durations:
        return None
    return max(durations.items(), key=lambda kv: kv[1])[0]

def transpose_delta_to_c_or_a(root_mode: Optional[Tuple[str, str]]) -> int:
    """Semitone shift so a major key becomes C-major or a minor key becomes A-minor.

    The shift is chosen in [-6, +5] so notes never move by more than half an octave.
    """
    if root_mode is None:
        return 0
    root, mode = root_mode
    pc = NOTE_TO_PC.get(root)
    if pc is None:
        return 0
    target = 9 if str(mode).lower().startswith("min") else 0
    delta = (target - pc) % 12
    if delta > 6:
        delta -= 12
    return delta

def transpose_chord_label(label: str, delta: int) -> str:
    label = normalize_chord_label(label)
    if label in ("N", "OOV", ""):
        return label
    if ":" in label:
        root, qual = label.split(":", 1)
    else:
        root, qual = label, "maj"
    pc = NOTE_TO_PC.get(root)
    if pc is None:
        return label
    return f"{_PC_TO_NOTE[(pc + delta) % 12]}:{qual}"

def parse_chord_file(path: str | Path) -> List[Tuple[float, float, str]]:
    path = Path(path)
    if not path.exists():
        return []
    vals = path.read_text(encoding="utf-8").strip().split()
    segs = []
    for i in range(0, len(vals)-2, 3):
        try:
            a, b, c = float(vals[i]), float(vals[i+1]), normalize_chord_label(vals[i+2])
            if b > a: segs.append((a, b, c))
        except ValueError:
            pass
    return segs

def load_song(song_dir: str | Path, normalize_key: bool = True) -> Optional[SongData]:
    song_dir = Path(song_dir); song_id = song_dir.name; midi_path = song_dir / f"{song_id}.mid"
    if not midi_path.exists(): return None
    melody = choose_melody_track(extract_notes_by_track(midi_path))
    beats, downs = parse_beat_file(song_dir / "beat_midi.txt")
    chords = parse_chord_file(song_dir / "chord_midi.txt")
    if not melody or len(beats) < 8 or not chords: return None
    key = parse_key_file(song_dir / "key_audio.txt") if normalize_key else None
    delta = transpose_delta_to_c_or_a(key) if normalize_key else 0
    if delta:
        melody = [Note(int(n.pitch) + delta, n.start, n.end, n.velocity) for n in melody]
        chords = [(a, b, transpose_chord_label(c, delta)) for (a, b, c) in chords]
    return SongData(song_id, midi_path, melody, beats, downs, chords, key=key, transpose_delta=delta)

def pitch_at_time(notes: Sequence[Note], t: float, default: int = REST) -> int:
    active = [n.pitch for n in notes if n.start <= t < n.end]
    return max(active) if active else default

def frame_melody(notes: Sequence[Note], total_beats: float, step: float = 0.5) -> List[int]:
    n_frames = max(1, int(math.ceil(total_beats / step)))
    return [pitch_at_time(notes, i*step + 0.5*step, REST) for i in range(n_frames)]

def normalize_chord_label(label: str) -> str:
    label = label.strip()
    if label in ("", "N", "X"): return "N"
    return label.split("/")[0]

def chord_at_time(segs: Sequence[Tuple[float,float,str]], t: float, default="N") -> str:
    for a,b,c in segs:
        if a <= t < b: return c
    return default

def chord_to_pitches(label: str, octave: int = 3) -> List[int]:
    label = normalize_chord_label(label)
    if label == "N": return []
    root, qual = (label.split(":", 1) + ["maj"])[:2] if ":" in label else (label, "maj")
    if root not in NOTE_TO_PC: return []
    if qual in QUALITY_INTERVALS: intervals = QUALITY_INTERVALS[qual]
    elif qual.startswith("min"): intervals = QUALITY_INTERVALS["min"]
    elif qual.startswith("maj"): intervals = QUALITY_INTERVALS["maj"]
    elif qual.startswith("dim"): intervals = QUALITY_INTERVALS["dim"]
    elif "sus4" in qual: intervals = QUALITY_INTERVALS["sus4"]
    elif "sus2" in qual: intervals = QUALITY_INTERVALS["sus2"]
    elif "7" in qual: intervals = QUALITY_INTERVALS["7"]
    else: intervals = QUALITY_INTERVALS["maj"]
    base = 12 * (octave + 1) + NOTE_TO_PC[root]
    return [base+i for i in intervals]

def melody_note_in_chord_rate(melody_pitches: Sequence[int], chord_labels: Sequence[str]) -> float:
    total = hit = 0
    for p,c in zip(melody_pitches, chord_labels):
        if p == REST: continue
        pcs = {x % 12 for x in chord_to_pitches(c)}
        if not pcs: continue
        total += 1; hit += int(p % 12 in pcs)
    return hit / total if total else 0.0

def write_midi_combined(path, melody_notes: Sequence[Note], chord_labels: Sequence[str],
                         beat_offset: float = 0.0, tempo: int = 105, style: str = "ballad"):
    """Write MIDI from explicit melody Notes (times in beats) + per-beat chord labels.

    The chord track always starts at beat 0 of the output. `beat_offset` is
    subtracted from every melody note's start/end, so passing the beat index of
    the first real melody note effectively trims the intro and aligns left and
    right hand. Notes that fall outside [0, len(chord_labels)] are kept (we just
    let melody overhang on either side rather than clipping musical phrases).
    """
    path = Path(path); mid = mido.MidiFile(ticks_per_beat=480); tpb = mid.ticks_per_beat
    tr_m = mido.MidiTrack(); mid.tracks.append(tr_m)
    tr_m.append(mido.MetaMessage("track_name", name="MELODY", time=0))
    tr_m.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    events = []
    for n in melody_notes:
        a = int(round((n.start - beat_offset) * tpb))
        b = int(round((n.end - beat_offset) * tpb))
        if b <= a:
            b = a + tpb // 4
        if b <= 0:
            continue
        a = max(0, a)
        events += [
            (a, mido.Message("note_on", note=int(n.pitch), velocity=int(n.velocity), time=0, channel=0)),
            (b, mido.Message("note_off", note=int(n.pitch), velocity=0, time=0, channel=0)),
        ]
    _write_events(tr_m, events)
    if chord_labels:
        tr_c = mido.MidiTrack(); mid.tracks.append(tr_c)
        tr_c.append(mido.MetaMessage("track_name", name=f"PIANO_{style}", time=0))
        _append_chords(tr_c, chord_labels, tpb, 1, style)
    path.parent.mkdir(parents=True, exist_ok=True); mid.save(path)

def write_midi_from_notes(path, notes: Sequence[Note], tempo: int = 105):
    """Write a melody track from explicit `Note` objects whose times are in beats.

    Used by the REMI-decoder path: the decoder produces notes with start/end in
    beat units, so we just multiply by ticks-per-beat.
    """
    path = Path(path); mid = mido.MidiFile(ticks_per_beat=480); tpb = mid.ticks_per_beat
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name="MELODY", time=0))
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    events = []
    for n in notes:
        a = int(round(n.start * tpb)); b = int(round(n.end * tpb))
        if b <= a:
            b = a + tpb // 4
        events += [
            (a, mido.Message("note_on", note=int(n.pitch), velocity=int(n.velocity), time=0, channel=0)),
            (b, mido.Message("note_off", note=int(n.pitch), velocity=0, time=0, channel=0)),
        ]
    _write_events(tr, events)
    path.parent.mkdir(parents=True, exist_ok=True); mid.save(path)

def write_midi(path, melody_frames=None, chord_labels=None, step=0.5, tempo=105, style="ballad"):
    path = Path(path); mid = mido.MidiFile(ticks_per_beat=480)
    if melody_frames is not None:
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name="MELODY", time=0)); tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
        _append_melody(tr, melody_frames, step, mid.ticks_per_beat, 0)
    if chord_labels is not None:
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=f"PIANO_{style}", time=0))
        _append_chords(tr, chord_labels, mid.ticks_per_beat, 1, style)
    path.parent.mkdir(parents=True, exist_ok=True); mid.save(path)

def _append_melody(track, frames, step, tpb, channel):
    events=[]; i=0
    while i < len(frames):
        p=frames[i]; j=i+1
        while j < len(frames) and frames[j] == p: j += 1
        if p != REST:
            a=int(round(i*step*tpb)); b=int(round(j*step*tpb))
            events += [(a,mido.Message("note_on",note=int(p),velocity=88,time=0,channel=channel)),(b,mido.Message("note_off",note=int(p),velocity=0,time=0,channel=channel))]
        i=j
    _write_events(track, events)

def _root_pc(label: str) -> Optional[int]:
    label = normalize_chord_label(label)
    if label == "N": return None
    root = label.split(":", 1)[0]
    return NOTE_TO_PC.get(root)

def _voice_led_chord(label: str, prev_upper: Optional[List[int]] = None, octave: int = 3) -> Tuple[List[int], List[int]]:
    """Return (bass_notes, upper_chord_notes) with simple voice leading.

    Earlier versions rendered every chord in fixed root position, which makes the left
    hand sound like the same shape repeated forever. This function tries inversions and
    neighboring octaves, choosing the voicing with the smallest movement from the
    previous upper chord while keeping a real bass root below it.
    """
    base = chord_to_pitches(label, octave=octave)
    if not base:
        return [], []
    root = _root_pc(label)
    # Build candidate upper voicings in a comfortable piano range.
    pcs = [p % 12 for p in base]
    candidates: List[List[int]] = []
    for low_oct in [3, 4]:
        notes = []
        for pc in pcs[:4]:
            n = 12 * (low_oct + 1) + pc
            while n < 55:
                n += 12
            while n > 76:
                n -= 12
            notes.append(n)
        # Try inversions by lifting lower notes an octave.
        notes = sorted(notes)
        for inv in range(min(4, len(notes))):
            cand = sorted([x + (12 if j < inv else 0) for j, x in enumerate(notes)])
            if max(cand) - min(cand) <= 18:
                candidates.append(cand)
    if not candidates:
        candidates = [base]
    if prev_upper:
        def dist(c):
            m = min(len(c), len(prev_upper))
            return sum(abs(c[i] - prev_upper[i]) for i in range(m)) + 0.25 * (max(c) - min(c))
        upper = min(candidates, key=dist)
    else:
        upper = min(candidates, key=lambda c: abs(sum(c)/len(c) - 64))
    if root is None:
        bass = [upper[0] - 12]
    else:
        b = 12 * 3 + root  # C2-ish root register under the chord
        while b > min(upper) - 14:
            b -= 12
        while b < 36:
            b += 12
        fifth = b + 7 if b + 7 < min(upper) - 3 else b - 5
        bass = [b, fifth]
    return bass, upper

def _append_chords(track, chords, tpb, channel, style):
    """Render chords. `style` is either a single style string applied to all beats,
    or a list of style strings of length `len(chords)` to vary the pattern per beat
    (useful for density-adaptive accompaniment)."""
    is_list = isinstance(style, (list, tuple))
    events=[]
    prev_upper: Optional[List[int]] = None
    for i,c in enumerate(chords):
        bass, upper = _voice_led_chord(c, prev_upper)
        if not upper and not bass:
            continue
        prev_upper = upper
        start = i * tpb
        beat_in_bar = i % 4
        # Slightly vary velocities so the accompaniment breathes.
        strong = beat_in_bar == 0
        bass_vel = 70 if strong else 58
        chord_vel = 58 if strong else 50
        cur_style = style[i] if is_list else style
        if cur_style == "block":
            # Root in bass + voice-led upper chord, not a fixed root-position triad.
            for n in bass[:1] + upper:
                vel = bass_vel if n in bass else chord_vel
                events += [(start,mido.Message("note_on",note=int(n),velocity=vel,time=0,channel=channel)),(start+int(.86*tpb),mido.Message("note_off",note=int(n),velocity=0,time=0,channel=channel))]
        elif cur_style == "arpeggio":
            pattern = [bass[0] if bass else upper[0]] + upper
            if i % 2 == 1:
                pattern = [pattern[0]] + list(reversed(pattern[1:]))
            sub = max(1, tpb // max(4, len(pattern)))
            for k,n in enumerate(pattern[:6]):
                st=start+k*sub
                events += [(st,mido.Message("note_on",note=int(n),velocity=66 if k==0 else 54,time=0,channel=channel)),(st+int(.9*sub),mido.Message("note_off",note=int(n),velocity=0,time=0,channel=channel))]
        elif cur_style == "syncopated":
            root = bass[0] if bass else upper[0]-12
            hits = [(0, [root], 68), (tpb//2, upper, 54), (3*tpb//4, upper[1:]+upper[:1], 48)]
            for offset, ns, vel in hits:
                dur = int(.38*tpb if offset else .45*tpb)
                for n in ns:
                    events += [(start+offset,mido.Message("note_on",note=int(n),velocity=vel,time=0,channel=channel)),(start+offset+dur,mido.Message("note_off",note=int(n),velocity=0,time=0,channel=channel))]
        elif cur_style == "ballad_light":
            # Half-density ballad: bass on beat, chord on the "and". 2 hits per beat
            # instead of 4. Alternates root and fifth on consecutive beats so the
            # bass line still moves without sounding mechanical.
            root = bass[0] if bass else upper[0]-12
            fifth = bass[1] if len(bass) > 1 else root + 7
            bass_note = root if i % 2 == 0 else fifth
            # Chord stab uses a 3-note upper voicing to avoid clashing with melody.
            chord_voicing = upper[:3]
            hits = [(0, [bass_note], bass_vel),
                    (tpb // 2, chord_voicing, chord_vel - 4)]
            for offset, ns, vel in hits:
                dur = int(0.45 * tpb)
                for n in ns:
                    events += [(start+offset, mido.Message("note_on", note=int(n), velocity=max(30, vel), time=0, channel=channel)),
                               (start+offset+dur, mido.Message("note_off", note=int(n), velocity=0, time=0, channel=channel))]
        elif cur_style == "arpeggio_light":
            # Half-density arpeggio: only fires on strong beats (beats 1 and 3 of
            # each bar). Silent beats let the melody breathe.
            if beat_in_bar in (0, 2):
                pattern = [bass[0] if bass else upper[0]-12] + upper[:3]
                sub = max(1, tpb // 3)  # three notes over the beat
                for k, n in enumerate(pattern[:3]):
                    st = start + k * sub
                    events += [(st, mido.Message("note_on", note=int(n), velocity=58 if k == 0 else 48, time=0, channel=channel)),
                               (st + int(0.9 * sub), mido.Message("note_off", note=int(n), velocity=0, time=0, channel=channel))]
        elif cur_style == "syncopated_light":
            # Bass on the downbeat, chord on the "and". 2 hits/beat.
            root_n = bass[0] if bass else upper[0]-12
            chord_voicing = upper[:3]
            for offset, ns, vel in [(0, [root_n], bass_vel),
                                    (tpb // 2, chord_voicing, chord_vel - 6)]:
                dur = int(0.42 * tpb)
                for n in ns:
                    events += [(start+offset, mido.Message("note_on", note=int(n), velocity=max(30, vel), time=0, channel=channel)),
                               (start+offset+dur, mido.Message("note_off", note=int(n), velocity=0, time=0, channel=channel))]
        else:  # ballad
            root = bass[0] if bass else upper[0]-12
            fifth = bass[1] if len(bass) > 1 else root + 7
            u = upper + upper[:2]
            # Pop-ballad pattern: bass, fifth, inner chord tone, top/inner. It changes
            # with inversions, so it should not sound like the same left hand forever.
            pattern = [root, fifth, u[0], u[2 % len(u)], u[1 % len(u)], u[2 % len(u)]]
            offsets = [0, tpb//4, tpb//2, 3*tpb//4, tpb, tpb + tpb//2]
            # Keep pattern within this beat most of the time; occasional anticipation
            # creates movement without disrupting the melody.
            offsets = [o for o in offsets if o < tpb]
            sub_dur = int(.24*tpb)
            for k,off in enumerate(offsets):
                n = pattern[k]
                events += [(start+off,mido.Message("note_on",note=int(n),velocity=68 if k==0 else 52,time=0,channel=channel)),(start+off+sub_dur,mido.Message("note_off",note=int(n),velocity=0,time=0,channel=channel))]
    _write_events(track, events)

def _write_events(track, events):
    events.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))
    last=0
    for tick,msg in events:
        msg.time=max(0,int(tick-last)); track.append(msg); last=tick
