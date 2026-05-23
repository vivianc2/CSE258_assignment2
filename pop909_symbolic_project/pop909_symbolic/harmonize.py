from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple
import math, random
import numpy as np
import torch
from .constants import REST, CHORD_N, CHORD_OOV
from .midi_io import chord_to_pitches, normalize_chord_label


def build_transition_logprobs(train_songs, chord_to_idx: Dict[str, int], alpha: float = 0.05) -> np.ndarray:
    """Smoothed chord-to-chord transition log-probabilities from training songs."""
    V = len(chord_to_idx)
    counts = np.full((V, V), alpha, dtype=np.float64)
    for s in train_songs:
        ids = [chord_to_idx.get(c, chord_to_idx.get(CHORD_OOV, 1)) for c in s.chord_labels]
        for a, b in zip(ids, ids[1:]):
            counts[a, b] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs + 1e-12)


def chord_pitch_class_set(label: str) -> set[int]:
    return {p % 12 for p in chord_to_pitches(label)}


def melody_fit_score(melody_pitch: int, label: str) -> float:
    """A soft music-theory score for whether a chord supports the melody note."""
    if melody_pitch == REST:
        return 0.0
    pcs = chord_pitch_class_set(label)
    if not pcs:
        return -0.8
    pc = melody_pitch % 12
    if pc in pcs:
        return 1.0
    # Passing-tone tolerance: seconds and fourths can still sound okay briefly.
    distances = {min((pc - c) % 12, (c - pc) % 12) for c in pcs}
    if min(distances) <= 2:
        return 0.15
    return -1.0


def _candidate_indices(logprob_row: np.ndarray, idx_to_chord: Sequence[str], top_k: int) -> List[int]:
    order = np.argsort(-logprob_row)
    out = []
    for i in order:
        lab = idx_to_chord[int(i)]
        if lab in (CHORD_N, CHORD_OOV):
            continue
        out.append(int(i))
        if len(out) >= top_k:
            break
    return out or [int(order[0])]


def guided_decode_chords(
    logits: torch.Tensor,
    melody_pitches: Sequence[int],
    idx_to_chord: Sequence[str],
    transition_logprobs: np.ndarray | None = None,
    top_k: int = 10,
    beam_size: int = 6,
    temperature: float = 0.9,
    model_weight: float = 1.0,
    fit_weight: float = 1.2,
    transition_weight: float = 0.55,
    repeat_penalty: float = 0.42,
    phrase_change_bonus: float = 0.32,
    max_same: int = 3,
    seed: int = 42,
) -> List[str]:
    """Decode model chord logits into a more musical, less repetitive chord sequence.

    The raw argmax from a chord classifier often collapses to the most common chord. This
    decoder keeps the model's probabilities but adds: melody-note compatibility, learned
    chord transition priors, phrase-level change pressure, and a penalty for repeating
    the same chord too long.
    """
    rng = random.Random(seed)
    if logits.dim() == 3:
        logits = logits[0]
    logprobs = torch.log_softmax(logits / max(temperature, 1e-6), dim=-1).detach().cpu().numpy()
    T, V = logprobs.shape
    T = min(T, len(melody_pitches))

    # Beam item: (score, sequence_indices, last_idx, run_length)
    beams: List[Tuple[float, List[int], int | None, int]] = [(0.0, [], None, 0)]
    for t in range(T):
        candidates = _candidate_indices(logprobs[t], idx_to_chord, top_k)
        new_beams: List[Tuple[float, List[int], int, int]] = []
        for score, seq, last, run in beams:
            for c in candidates:
                label = idx_to_chord[c]
                new_run = run + 1 if last == c else 1
                if new_run > max_same:
                    # Not impossible, but make it very unattractive.
                    hard_repeat_cost = -2.5 * (new_run - max_same)
                else:
                    hard_repeat_cost = 0.0
                s = score
                s += model_weight * float(logprobs[t, c])
                s += fit_weight * melody_fit_score(int(melody_pitches[t]), label)
                if last is not None and transition_logprobs is not None:
                    s += transition_weight * float(transition_logprobs[last, c])
                if last == c:
                    s -= repeat_penalty * max(0, new_run - 1)
                else:
                    # Encourage changes at phrase-ish places, but don't force chaos.
                    if t % 4 == 0 or t % 8 == 0:
                        s += phrase_change_bonus
                # Tiny deterministic jitter to break ties without making the run unstable.
                s += 1e-5 * rng.random()
                s += hard_repeat_cost
                new_beams.append((s, seq + [c], c, new_run))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]
    best = beams[0][1]
    return [idx_to_chord[i] for i in best]


def summarize_chord_sequence(chords: Sequence[str]) -> Dict[str, float | int | str]:
    if not chords:
        return {"n_chords": 0, "unique_chords": 0, "change_rate": 0.0, "preview": ""}
    changes = sum(a != b for a, b in zip(chords, chords[1:]))
    return {
        "n_chords": len(chords),
        "unique_chords": len(set(chords)),
        "change_rate": changes / max(1, len(chords) - 1),
        "preview": " ".join(chords[:32]),
    }
