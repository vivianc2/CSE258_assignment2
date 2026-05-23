"""Re-sample the REMI Transformer melody LM from an existing checkpoint.

No training. Useful for sweeping sampling knobs (temperature, top-p, repetition
penalty) and producing a variety of melodies for the presentation.

Example — produce three samples with different temperatures:

    for t in 0.8 1.0 1.2; do
      python resample_remi.py --out_dir outputs_final --temperature $t \\
          --tag t$t --bars 32
    done

The resulting MIDIs are written to `<out_dir>/samples/remi_<tag>.mid`.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from pop909_symbolic.models import MelodyTransformerLM
from pop909_symbolic.training import sample_remi
from pop909_symbolic.midi_io import write_midi_from_notes
from pop909_symbolic import remi as RR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs_final")
    ap.add_argument("--tag", default="resampled",
                    help="Suffix for output filename: <out_dir>/samples/remi_<tag>.mid")
    ap.add_argument("--bars", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--repeat_penalty", type=float, default=1.15)
    ap.add_argument("--tempo", type=int, default=105)
    ap.add_argument("--seed", type=int, default=None,
                    help="If omitted, samples a fresh seed each run.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir)
    ck = torch.load(out / "remi_melody_transformer.pt", map_location=args.device,
                    weights_only=False)
    ra = ck["args"]
    model = MelodyTransformerLM(
        d_model=ra.get("remi_d_model", 384), nhead=ra.get("remi_nhead", 6),
        num_layers=ra.get("remi_layers", 6), dim_feedforward=ra.get("remi_ff", 1536),
        dropout=ra.get("remi_dropout", 0.1),
        max_len=max(ra.get("remi_seq_len", 384), 1024),
    ).to(args.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    tokens = sample_remi(
        model, n_bars=args.bars, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        pitch_repeat_penalty=args.repeat_penalty,
        device=args.device, seed=args.seed,
    )
    decoded = RR.decode_remi(tokens)
    notes = RR.decoded_notes_to_midi_notes(decoded, tempo_bpm=args.tempo)

    samples_dir = out / "samples"
    samples_dir.mkdir(exist_ok=True)
    mid_path = samples_dir / f"remi_{args.tag}.mid"
    write_midi_from_notes(mid_path, notes, tempo=args.tempo)
    tok_path = samples_dir / f"remi_{args.tag}.tokens.txt"
    tok_path.write_text(" ".join(str(t) for t in tokens))
    meta = {
        "tag": args.tag, "n_tokens": len(tokens), "n_notes": len(notes),
        "bars": args.bars, "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k, "repeat_penalty": args.repeat_penalty,
        "seed": args.seed,
    }
    (samples_dir / f"remi_{args.tag}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {mid_path}  ({len(notes)} notes)")
    print(f"      {tok_path}")


if __name__ == "__main__":
    main()
