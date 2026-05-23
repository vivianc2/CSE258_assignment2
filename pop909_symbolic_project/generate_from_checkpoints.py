"""Generate fresh MIDIs from existing checkpoints without retraining.

This is a quick wrapper: it loads the REMI Transformer (Task 1 primary model)
and the chord Transformer (Task 2 primary model) and writes fresh samples. For
the polished Task 2 render (natural melody timing, intro-aligned chords,
self-harmonized REMI melody), use `render_better_outputs.py` instead.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from pop909_symbolic.midi_io import (
    list_song_dirs, load_song, write_midi, write_midi_from_notes,
)
from pop909_symbolic.datasets import prepare_song
from pop909_symbolic.models import MelodyLSTM, MelodyTransformerLM, ChordTransformer
from pop909_symbolic.training import sample_melody, sample_remi, predict_chords
from pop909_symbolic.harmonize import build_transition_logprobs
from pop909_symbolic.constants import REST
from pop909_symbolic import remi as RR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop909_root", required=True)
    ap.add_argument("--out_dir", default="outputs_final")
    ap.add_argument("--song_index", type=int, default=0)
    ap.add_argument("--tempo", type=int, default=105)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--also_lstm", action="store_true",
                    help="Additionally sample from the LSTM baseline (for A/B comparison).")
    args = ap.parse_args()
    out = Path(args.out_dir)

    # --- REMI Transformer melody sample (Task 1 primary) ----------
    remi_ckpt = out / "remi_melody_transformer.pt"
    if remi_ckpt.exists():
        ck = torch.load(remi_ckpt, map_location=args.device, weights_only=False)
        ra = ck["args"]
        rem = MelodyTransformerLM(
            d_model=ra.get("remi_d_model", 384), nhead=ra.get("remi_nhead", 6),
            num_layers=ra.get("remi_layers", 6), dim_feedforward=ra.get("remi_ff", 1536),
            dropout=ra.get("remi_dropout", 0.1),
            max_len=max(ra.get("remi_seq_len", 384), 1024),
        ).to(args.device)
        rem.load_state_dict(ck["model_state"])
        toks = sample_remi(rem, n_bars=ra.get("remi_gen_bars", 32),
                           temperature=ra.get("remi_temperature", 1.0),
                           top_p=ra.get("remi_top_p", 0.92),
                           top_k=ra.get("remi_top_k", 0),
                           pitch_repeat_penalty=ra.get("remi_repeat_penalty", 1.15),
                           device=args.device, seed=ra.get("seed", 42) + 1)
        notes = RR.decoded_notes_to_midi_notes(RR.decode_remi(toks), tempo_bpm=args.tempo)
        write_midi_from_notes(out / "new_symbolic_unconditioned.mid", notes, tempo=args.tempo)
        print(f"REMI Transformer: wrote {out/'new_symbolic_unconditioned.mid'} ({len(notes)} notes)")
    else:
        print("No REMI checkpoint found at", remi_ckpt)

    # --- LSTM baseline melody sample (optional, for A/B) ----------
    if args.also_lstm:
        lstm_ckpt = out / "melody_lstm.pt"
        if lstm_ckpt.exists():
            ck = torch.load(lstm_ckpt, map_location=args.device, weights_only=False)
            la = ck["args"]
            mel = MelodyLSTM(
                emb_dim=la.get("melody_emb", 96),
                hidden_dim=la.get("melody_hidden", 192),
                num_layers=la.get("melody_layers", 2),
            ).to(args.device)
            mel.load_state_dict(ck["model_state"])
            frames = sample_melody(mel, length=la.get("gen_melody_frames", 192),
                                   temperature=la.get("temperature", 0.9),
                                   top_k=la.get("top_k", 12),
                                   seed_token=REST, device=args.device)
            write_midi(out / "new_baseline_lstm_melody.mid", melody_frames=frames,
                       step=la.get("frame_step", 0.5), tempo=args.tempo)
            print(f"LSTM baseline:    wrote {out/'new_baseline_lstm_melody.mid'}")

    # --- Chord prediction on a held-out POP909 melody --------------
    chord_ckpt = out / "chord_transformer.pt"
    if chord_ckpt.exists():
        ck2 = torch.load(chord_ckpt, map_location=args.device, weights_only=False)
        ca = ck2["args"]
        idx_to_chord = ck2["idx_to_chord"]; chord_to_idx = ck2["chord_to_idx"]
        ch = ChordTransformer(
            len(idx_to_chord), d_model=ca.get("chord_dim", 128),
            nhead=ca.get("chord_heads", 4), num_layers=ca.get("chord_layers", 3),
            max_len=ca.get("max_beats", 256),
        ).to(args.device)
        ch.load_state_dict(ck2["model_state"])

        songs = []
        for d in list_song_dirs(args.pop909_root):
            s = load_song(d)
            if s is not None:
                songs.append(s)
        prepared = [prepare_song(s) for s in songs]
        song = prepared[args.song_index % len(prepared)]
        transition_logprobs = build_transition_logprobs(
            prepared[: min(200, len(prepared))], chord_to_idx,
        )
        import types
        decode_ns = types.SimpleNamespace(
            decode_strategy=ca.get("decode_strategy", "guided"),
            chord_top_k=ca.get("chord_top_k", 10),
            chord_beam_size=ca.get("chord_beam_size", 6),
            chord_temperature=ca.get("chord_temperature", 0.9),
            decode_model_weight=ca.get("decode_model_weight", 1.0),
            decode_fit_weight=ca.get("decode_fit_weight", 1.2),
            decode_transition_weight=ca.get("decode_transition_weight", 0.55),
            decode_repeat_penalty=ca.get("decode_repeat_penalty", 0.42),
            decode_phrase_change_bonus=ca.get("decode_phrase_change_bonus", 0.32),
            max_same_chord=ca.get("max_same_chord", 3),
            seed=ca.get("seed", 42),
        )
        chords, bm, _ = predict_chords(ch, song, chord_to_idx, idx_to_chord,
                                       args.device, 96,
                                       transition_logprobs=transition_logprobs,
                                       args=decode_ns)
        mf = []
        for p in bm: mf.extend([p, p])
        write_midi(out / "new_symbolic_conditioned.mid", melody_frames=mf,
                   chord_labels=chords, step=0.5, tempo=args.tempo, style="ballad")
        print(f"Chord Transformer: wrote {out/'new_symbolic_conditioned.mid'}")
        print("NOTE: this uses the basic beat-grid melody. For a polished render with")
        print("      natural rhythm + intro alignment + self-harmonized Transformer melody,")
        print("      use `render_better_outputs.py` instead.")


if __name__ == "__main__":
    main()
