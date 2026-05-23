"""Re-render Task 2 outputs from already-trained checkpoints — no retraining.

Fixes two problems with the existing `outputs_final/symbolic_conditioned.mid`:

1. The prompt song had a long instrumental intro, so chords played for many bars
   before the melody started. We now pick a prompt with melody close to beat 0,
   and shift the chord track to align with the first melody note.

2. The right hand was a single pitch held per beat (one sample per beat from the
   prompt song). We now render the original melody notes with their natural
   timing, which preserves the song's rhythm and is much denser.

Plus a fun extra: harmonize the REMI-Transformer-sampled melody end-to-end. This
ties Task 1 and Task 2 together — every note of pitch *and* every chord came
from our learned models.

Usage:

    python render_better_outputs.py --out_dir outputs_final --pop909_root ../POP909-Dataset/POP909
"""
from __future__ import annotations
import argparse, json, math, types
from pathlib import Path
from typing import List, Optional
import shutil
import torch

from pop909_symbolic.constants import REST
from pop909_symbolic.midi_io import (
    Note, list_song_dirs, load_song, write_midi_combined, write_midi_from_notes,
)
from pop909_symbolic.datasets import PreparedSong, prepare_song
from pop909_symbolic.models import MelodyTransformerLM, ChordTransformer
from pop909_symbolic.training import predict_chords, sample_remi
from pop909_symbolic.harmonize import build_transition_logprobs
from pop909_symbolic import remi as RR


def find_good_prompt(songs, max_intro_beats: int = 6, min_melody_notes: int = 24):
    """Pick a prompt song whose melody starts close to beat 0 and has enough notes."""
    candidates = []
    for s in songs:
        if not s.melody_notes or len(s.melody_notes) < min_melody_notes:
            continue
        intro = float(s.melody_notes[0].start)
        candidates.append((intro, len(s.melody_notes), s))
    candidates.sort(key=lambda x: (x[0], -x[1]))  # shortest intro, then most notes
    for intro, _, s in candidates:
        if intro <= max_intro_beats:
            return s
    return candidates[0][2] if candidates else None


def quantize_melody(notes: List[Note], grid_beats: float = 0.5,
                    min_dur_beats: float = 0.5) -> List[Note]:
    """Snap every note's start and end to a fixed beat grid.

    POP909 melodies have expressive (live-played) timing — a note "on the
    downbeat" might land at 0.42 beats, which clashes with chord hits on the
    strict 0.0/0.5/1.0 grid. Snapping to the nearest `grid_beats` step makes
    melody and accompaniment line up rhythmically.

    Also enforces a minimum duration so very short notes don't slip between
    grid positions and disappear after dedup.
    """
    if grid_beats <= 0:
        return list(notes)
    out: List[Note] = []
    for n in notes:
        s = round(n.start / grid_beats) * grid_beats
        e = round(n.end / grid_beats) * grid_beats
        if e - s < min_dur_beats:
            e = s + min_dur_beats
        out.append(Note(int(n.pitch), float(s), float(e), int(n.velocity)))
    out.sort(key=lambda n: (n.start, -n.pitch))
    # After snapping, collisions at the same start are common — keep the higher
    # pitch (melody convention) and extend its duration through the collision.
    deduped: List[Note] = []
    for n in out:
        if deduped and abs(deduped[-1].start - n.start) < 1e-6:
            prev = deduped[-1]
            keep = n if n.pitch >= prev.pitch else prev
            deduped[-1] = Note(int(keep.pitch), prev.start, max(prev.end, n.end),
                                int(keep.velocity))
            continue
        deduped.append(n)
    return deduped


def thin_melody(notes: List[Note], min_spacing_beats: float = 0.375,
                 keep_higher: bool = True) -> List[Note]:
    """Drop notes that crowd onto the same beat region so the melody isn't a wall of 16ths.

    Two notes within `min_spacing_beats` (default ~3/8 of a beat, i.e. close to
    a sixteenth-note) are collapsed: we keep whichever is higher (the melodic
    contour usually lives in the top voice) and extend it through the dropped
    note's duration.
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: (n.start, -n.pitch))
    kept: List[Note] = []
    for n in sorted_notes:
        if kept and (n.start - kept[-1].start) < min_spacing_beats:
            prev = kept[-1]
            # Choose the louder voice; default to the higher pitch.
            if keep_higher and n.pitch > prev.pitch:
                # Replace prev with current but extend duration to cover both.
                new_end = max(prev.end, n.end)
                kept[-1] = Note(int(n.pitch), prev.start, new_end, int(n.velocity))
            else:
                kept[-1] = Note(int(prev.pitch), prev.start, max(prev.end, n.end), int(prev.velocity))
            continue
        kept.append(n)
    return kept


def melody_notes_per_beat(notes: List[Note], beat_start: int, beat_end: int) -> int:
    """Count melody notes whose onset falls in [beat_start, beat_end)."""
    return sum(1 for n in notes if beat_start <= n.start < beat_end)


def adaptive_chord_styles(notes: List[Note], n_beats: int, beat_offset: int,
                           bar_len: int = 4, busy_threshold: int = 6,
                           medium_threshold: int = 3) -> List[str]:
    """Pick a chord style per *beat* based on local melody density per *bar*.

    Returns a list of style strings of length `n_beats`. All beats inside the
    same bar get the same style, so the accompaniment doesn't change character
    mid-bar (which would sound jumpy). The chooser:

      melody-notes-in-bar > busy_threshold   -> "block"        (sustained, gets out of the way)
      busy_threshold >= notes > medium_th... -> "ballad_light" (2 hits per beat)
      notes <= medium_threshold              -> "ballad"       (denser 4-hits-per-beat)
    """
    styles = []
    for b in range(n_beats):
        bar_idx = b // bar_len
        bar_start = beat_offset + bar_idx * bar_len
        bar_end = bar_start + bar_len
        density = melody_notes_per_beat(notes, bar_start, bar_end)
        if density > busy_threshold:
            styles.append("block")
        elif density > medium_threshold:
            styles.append("ballad_light")
        else:
            styles.append("ballad")
    return styles


def args_for_decoding(d_args: dict) -> types.SimpleNamespace:
    """Reconstruct a namespace with the decoding knobs predict_chords expects."""
    ns = types.SimpleNamespace(
        decode_strategy=d_args.get("decode_strategy", "guided"),
        chord_top_k=d_args.get("chord_top_k", 10),
        chord_beam_size=d_args.get("chord_beam_size", 6),
        chord_temperature=d_args.get("chord_temperature", 0.9),
        decode_model_weight=d_args.get("decode_model_weight", 1.0),
        decode_fit_weight=d_args.get("decode_fit_weight", 1.2),
        decode_transition_weight=d_args.get("decode_transition_weight", 0.55),
        decode_repeat_penalty=d_args.get("decode_repeat_penalty", 0.42),
        decode_phrase_change_bonus=d_args.get("decode_phrase_change_bonus", 0.32),
        max_same_chord=d_args.get("max_same_chord", 3),
        seed=d_args.get("seed", 42),
    )
    return ns


def harmonize_transformer_melody(remi_model, chord_model, chord_to_idx, idx_to_chord,
                                 transition_logprobs, device: str, n_bars: int, tempo: int,
                                 seed: int, decode_args, top_p: float, temperature: float,
                                 repeat_penalty: float, top_k: int):
    """Sample a melody from the REMI Transformer, then harmonize it with the chord model."""
    tokens = sample_remi(remi_model, n_bars=n_bars, temperature=temperature,
                        top_k=top_k, top_p=top_p, pitch_repeat_penalty=repeat_penalty,
                        device=device, seed=seed)
    decoded = RR.decode_remi(tokens)
    if not decoded:
        return None
    notes = RR.decoded_notes_to_midi_notes(decoded, tempo_bpm=tempo)
    total_beats = int(math.ceil(max(n.end for n in notes))) + 1
    beat_melody, beat_positions = [], []
    for i in range(total_beats):
        mid_t = i + 0.5
        active = [n.pitch for n in notes if n.start <= mid_t < n.end]
        beat_melody.append(max(active) if active else REST)
        beat_positions.append(i % 8)
    ps = PreparedSong(
        song_id="generated",
        melody_frames=[],
        beat_melody=beat_melody,
        beat_positions=beat_positions,
        chord_labels=[""] * total_beats,
        remi_tokens=tokens,
    )
    chords, _, raw_chords = predict_chords(
        chord_model, ps, chord_to_idx, idx_to_chord, device, total_beats,
        transition_logprobs=transition_logprobs, args=decode_args,
    )
    return {"notes": notes, "chords": chords, "raw_chords": raw_chords, "tokens": tokens,
            "n_beats": total_beats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs_final",
                    help="Directory containing trained checkpoints (and where re-rendered MIDIs are written).")
    ap.add_argument("--pop909_root", required=True)
    ap.add_argument("--max_intro_beats", type=int, default=6,
                    help="Prefer prompt songs whose melody starts within this many beats.")
    ap.add_argument("--gen_chord_beats", type=int, default=96)
    ap.add_argument("--tempo", type=int, default=105)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self_harmonize_bars", type=int, default=24)
    ap.add_argument("--remi_top_p", type=float, default=0.92)
    ap.add_argument("--remi_temperature", type=float, default=1.0)
    ap.add_argument("--remi_top_k", type=int, default=0)
    ap.add_argument("--remi_repeat_penalty", type=float, default=1.15)
    ap.add_argument("--no_self_harmonize", action="store_true")
    # Density controls. Tune these to taste.
    ap.add_argument("--thin_spacing", type=float, default=0.375,
                    help="Minimum gap (in beats) between consecutive melody notes. "
                         "Larger = thinner right hand. 0.5=8ths only, 0.25=keeps 16ths.")
    ap.add_argument("--busy_threshold", type=int, default=6,
                    help="Bars with more melody notes than this get 'block' (sustained) accompaniment.")
    ap.add_argument("--medium_threshold", type=int, default=3,
                    help="Bars with notes in (medium_threshold, busy_threshold] get 'ballad_light' (2 hits/beat).")
    ap.add_argument("--quantize_grid", type=float, default=0.5,
                    help="Snap melody onsets/durations to this beat grid. 0.5=8ths, 0.25=16ths, "
                         "1.0=quarter notes. Set to 0 to disable (use natural expressive timing).")
    ap.add_argument("--prompt_song", default=None,
                    help="Force a specific POP909 song id (e.g. '001') instead of auto-picking.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if not out.exists():
        raise FileNotFoundError(f"out_dir {out} does not exist")

    # --- Load chord model ---
    ck = torch.load(out / "chord_transformer.pt", map_location=args.device, weights_only=False)
    saved = ck["args"]
    decode_ns = args_for_decoding(saved)
    idx_to_chord = ck["idx_to_chord"]
    chord_to_idx = ck["chord_to_idx"]
    chord_model = ChordTransformer(
        len(idx_to_chord), d_model=saved.get("chord_dim", 128),
        nhead=saved.get("chord_heads", 4), num_layers=saved.get("chord_layers", 3),
        max_len=saved.get("max_beats", 256),
    ).to(args.device)
    chord_model.load_state_dict(ck["model_state"])
    chord_model.eval()

    # --- Load REMI Transformer (optional) ---
    remi_model: Optional[MelodyTransformerLM] = None
    remi_ckpt = out / "remi_melody_transformer.pt"
    if remi_ckpt.exists():
        ck2 = torch.load(remi_ckpt, map_location=args.device, weights_only=False)
        ra = ck2["args"]
        remi_model = MelodyTransformerLM(
            d_model=ra.get("remi_d_model", 384), nhead=ra.get("remi_nhead", 6),
            num_layers=ra.get("remi_layers", 6), dim_feedforward=ra.get("remi_ff", 1536),
            dropout=ra.get("remi_dropout", 0.1),
            max_len=max(ra.get("remi_seq_len", 384), 1024),
        ).to(args.device)
        remi_model.load_state_dict(ck2["model_state"])
        remi_model.eval()

    # --- Find a prompt song with short intro ---
    dirs = list_song_dirs(args.pop909_root)
    songs = []
    for d in dirs:
        s = load_song(d)
        if s is not None:
            songs.append(s)
    if not songs:
        raise RuntimeError("No POP909 songs found.")
    if args.prompt_song:
        matches = [s for s in songs if s.song_id == args.prompt_song]
        if not matches:
            raise RuntimeError(f"Prompt song {args.prompt_song!r} not found.")
        prompt = matches[0]
    else:
        prompt = find_good_prompt(songs, max_intro_beats=args.max_intro_beats)
    if prompt is None:
        raise RuntimeError("Could not find a usable prompt song.")
    intro_beats = float(prompt.melody_notes[0].start)
    print(f"Prompt song: {prompt.song_id}, intro = {intro_beats:.2f} beats, "
          f"{len(prompt.melody_notes)} melody notes, key={prompt.key}")

    # --- Build chord predictions for the prompt ---
    transition_logprobs = build_transition_logprobs(
        [prepare_song(s) for s in songs[: min(200, len(songs))]],  # transition prior
        chord_to_idx,
    )
    prepared = prepare_song(prompt)
    chords, _, raw_chords = predict_chords(
        chord_model, prepared, chord_to_idx, idx_to_chord, args.device,
        args.gen_chord_beats, transition_logprobs=transition_logprobs, args=decode_ns,
    )

    # --- Aligned natural-melody renders -----------------------------------
    # Beat-index of the first melody note rounded down; we subtract this from
    # the chord track's effective offset by trimming melody notes (the chord
    # track always starts at beat 0 of the output).
    first_beat = int(math.floor(intro_beats))
    # Use the chord slice predicted from beat_melody[first_beat:]; if the slice
    # is shorter than 16 beats fall back to the head of chord_labels.
    chord_slice = chords[first_beat: first_beat + args.gen_chord_beats]
    if len(chord_slice) < 16:
        chord_slice = chords[: args.gen_chord_beats]
    last_beat = first_beat + len(chord_slice)
    notes_in = [n for n in prompt.melody_notes if first_beat <= n.start < last_beat + 2]
    raw_slice = raw_chords[first_beat: first_beat + args.gen_chord_beats]
    if len(raw_slice) < 16:
        raw_slice = raw_chords[: args.gen_chord_beats]

    # --- Quantize melody onsets/durations to a fixed beat grid so they line up
    # with chord hits (which fire on the strict 0/0.5/1/1.5... grid). Without
    # this, POP909's expressive live-played timing sounds drunk against rigid
    # accompaniment.
    quantized = quantize_melody(notes_in, grid_beats=args.quantize_grid,
                                 min_dur_beats=max(args.quantize_grid, 0.25))
    # --- Then thin: drop 16th-note ornaments so the melody has room to breathe.
    thinned = thin_melody(quantized, min_spacing_beats=args.thin_spacing,
                         keep_higher=True)
    print(f"Melody density: {len(notes_in)} natural -> {len(quantized)} quantized "
          f"-> {len(thinned)} thinned  (grid={args.quantize_grid}, min_spacing={args.thin_spacing})")

    # Per-bar adaptive style: choose pattern based on melody density per bar.
    # Block fills dense bars (gets out of the way), ballad_light fills medium,
    # full ballad fills sparse bars.
    adaptive_styles = adaptive_chord_styles(
        thinned, n_beats=len(chord_slice), beat_offset=first_beat,
        bar_len=4, busy_threshold=args.busy_threshold,
        medium_threshold=args.medium_threshold,
    )
    style_counts = {s: adaptive_styles.count(s) for s in set(adaptive_styles)}
    print(f"Adaptive style mix (per beat): {style_counts}")

    new_files = {}
    # 1) The submission: thinned melody + adaptive ballad/light/block per bar.
    p_main = out / "symbolic_conditioned_adaptive.mid"
    write_midi_combined(p_main, thinned, chord_slice, beat_offset=first_beat,
                       tempo=args.tempo, style=adaptive_styles)
    new_files["adaptive"] = p_main
    # 2) Fixed-style variants for the presentation A/B grid. Light styles use
    #    the thinned melody so the texture is consistent.
    for style in ("block", "ballad_light", "arpeggio_light", "syncopated_light"):
        p = out / f"symbolic_conditioned_natural_{style}.mid"
        write_midi_combined(p, thinned, chord_slice, beat_offset=first_beat,
                           tempo=args.tempo, style=style)
        new_files[style] = p
    # 3) Keep the original ballad render (untrimmed melody, busy accompaniment)
    #    as a "before" example for the presentation.
    write_midi_combined(out / "symbolic_conditioned_natural_ballad_busy.mid",
                       notes_in, chord_slice, beat_offset=first_beat,
                       tempo=args.tempo, style="ballad")
    # Diagnostic: raw argmax chords + thinned melody (presentation A/B contrast).
    write_midi_combined(out / "debug_raw_argmax_natural.mid", thinned, raw_slice,
                       beat_offset=first_beat, tempo=args.tempo, style="block")
    # Melody-only prompts: natural + thinned (so graders can A/B the right hand).
    write_midi_from_notes(out / "melody_prompt_natural.mid",
                         [Note(n.pitch, n.start - first_beat, n.end - first_beat, n.velocity)
                          for n in notes_in],
                         tempo=args.tempo)
    write_midi_from_notes(out / "melody_prompt_thinned.mid",
                         [Note(n.pitch, n.start - first_beat, n.end - first_beat, n.velocity)
                          for n in thinned],
                         tempo=args.tempo)

    # Promote the adaptive variant to the official submission file.
    shutil.copyfile(new_files["adaptive"], out / "symbolic_conditioned.mid")
    shutil.copyfile(out / "symbolic_conditioned.mid", Path("symbolic_conditioned.mid"))
    print("Wrote:")
    for k, v in new_files.items():
        print(f"  {v}")
    print(f"  {out/'symbolic_conditioned.mid'}  (== adaptive, top-level copy updated)")

    # --- End-to-end: harmonize the Transformer-sampled melody --------------
    if remi_model is not None and not args.no_self_harmonize:
        res = harmonize_transformer_melody(
            remi_model, chord_model, chord_to_idx, idx_to_chord,
            transition_logprobs, args.device, n_bars=args.self_harmonize_bars,
            tempo=args.tempo, seed=args.seed, decode_args=decode_ns,
            top_p=args.remi_top_p, temperature=args.remi_temperature,
            repeat_penalty=args.remi_repeat_penalty, top_k=args.remi_top_k,
        )
        if res is not None:
            sh_quantized = quantize_melody(
                res["notes"], grid_beats=args.quantize_grid,
                min_dur_beats=max(args.quantize_grid, 0.25),
            )
            sh_thinned = thin_melody(sh_quantized,
                                      min_spacing_beats=args.thin_spacing,
                                      keep_higher=True)
            sh_styles = adaptive_chord_styles(
                sh_thinned, n_beats=len(res["chords"]), beat_offset=0,
                bar_len=4, busy_threshold=args.busy_threshold,
                medium_threshold=args.medium_threshold,
            )
            # Adaptive (the submission-style render).
            p_adapt = out / "self_harmonized_adaptive.mid"
            write_midi_combined(p_adapt, sh_thinned, res["chords"],
                               tempo=args.tempo, style=sh_styles)
            print(f"  {p_adapt}  (self-harmonized, adaptive per-bar accompaniment)")
            for style in ("block", "ballad_light", "arpeggio_light"):
                p = out / f"self_harmonized_{style}.mid"
                write_midi_combined(p, sh_thinned, res["chords"],
                                   tempo=args.tempo, style=style)
                print(f"  {p}  (self-harmonized: melody from REMI Transformer, chords from ChordTransformer)")
            # Save metadata.
            meta = {"n_notes": len(res["notes"]), "n_beats": res["n_beats"],
                    "chord_preview": " ".join(res["chords"][:32]),
                    "raw_chord_preview": " ".join(res["raw_chords"][:32])}
            (out / "self_harmonized_summary.json").write_text(json.dumps(meta, indent=2))

    print("\nDone. Listen first to:")
    print(f"  {out}/symbolic_conditioned.mid                       (Task 2 submission: thinned melody + adaptive accompaniment)")
    print(f"  {out}/symbolic_conditioned_natural_ballad_busy.mid   (the OLD version with busy melody + busy left hand)")
    print(f"  {out}/debug_raw_argmax_natural.mid                   (raw classifier chords, for the 'why guided decoder' demo)")
    if remi_model is not None and not args.no_self_harmonize:
        print(f"  {out}/self_harmonized_adaptive.mid                  (end-to-end: REMI melody + adaptive accompaniment)")


if __name__ == "__main__":
    main()
