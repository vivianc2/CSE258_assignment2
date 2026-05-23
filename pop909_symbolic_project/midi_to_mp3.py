from __future__ import annotations
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path

DEFAULT_SF2_CANDIDATES = [
    Path("/home/ubuntu/miniconda3/envs/musicgen/lib/python3.10/site-packages/pretty_midi/TimGM6mb.sf2"),
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/TimGM6mb.sf2"),
    Path("/usr/share/soundfonts/default.sf2"),
]

def find_soundfont(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path).expanduser().resolve()
        if not p.exists():
            sys.exit(f"Soundfont not found: {p}")
        return p
    try:
        import pretty_midi  # noqa: F401
        import importlib.resources as ir
        for f in ir.files("pretty_midi").iterdir():
            if f.name.endswith(".sf2"):
                return Path(str(f))
    except Exception:
        pass
    for p in DEFAULT_SF2_CANDIDATES:
        if p.exists():
            return p
    sys.exit("No soundfont (.sf2) found. Pass --soundfont path/to/font.sf2.")

def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"Required tool '{name}' not on PATH. Activate the musicgen conda env or install it.")
    return path

def midi_to_wav(midi: Path, wav: Path, sf2: Path, sample_rate: int, gain: float) -> None:
    fluidsynth = require_tool("fluidsynth")
    cmd = [fluidsynth, "-ni", "-F", str(wav), "-r", str(sample_rate), "-g", str(gain), str(sf2), str(midi)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav.exists():
        sys.exit(f"fluidsynth failed for {midi}:\n{proc.stderr}")

def wav_to_mp3(wav: Path, mp3: Path, bitrate: str) -> None:
    ffmpeg = require_tool("ffmpeg")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not mp3.exists():
        sys.exit(f"ffmpeg failed for {wav}:\n{proc.stderr}")

def convert_one(midi: Path, out_dir: Path | None, sf2: Path, sample_rate: int, gain: float, bitrate: str, keep_wav: bool) -> Path:
    target_dir = out_dir if out_dir else midi.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = target_dir / f"{midi.stem}.mp3"
    if keep_wav:
        wav_path = target_dir / f"{midi.stem}.wav"
        midi_to_wav(midi, wav_path, sf2, sample_rate, gain)
        wav_to_mp3(wav_path, mp3_path, bitrate)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / f"{midi.stem}.wav"
            midi_to_wav(midi, wav_path, sf2, sample_rate, gain)
            wav_to_mp3(wav_path, mp3_path, bitrate)
    return mp3_path

def collect_inputs(inputs: list[str], recursive: bool) -> list[Path]:
    midis: list[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            pattern = "**/*.mid" if recursive else "*.mid"
            midis.extend(sorted(p.glob(pattern)))
            pattern_alt = "**/*.midi" if recursive else "*.midi"
            midis.extend(sorted(p.glob(pattern_alt)))
        elif p.is_file():
            midis.append(p)
        else:
            sys.exit(f"Input not found: {p}")
    seen, unique = set(), []
    for m in midis:
        r = m.resolve()
        if r not in seen:
            seen.add(r); unique.append(m)
    if not unique:
        sys.exit("No .mid/.midi files found in inputs.")
    return unique

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render MIDI files to MP3 via FluidSynth + ffmpeg")
    ap.add_argument("inputs", nargs="+", help="MIDI file(s) or directory(ies) to convert")
    ap.add_argument("--out_dir", default=None, help="Directory for MP3 output (default: alongside each input)")
    ap.add_argument("--soundfont", default=None, help="Path to .sf2 (defaults to the one bundled with pretty_midi)")
    ap.add_argument("--sample_rate", type=int, default=44100)
    ap.add_argument("--gain", type=float, default=0.6, help="FluidSynth synth gain (0.0-10.0)")
    ap.add_argument("--bitrate", default="192k", help="MP3 bitrate (e.g. 128k, 192k, 320k)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when an input is a directory")
    ap.add_argument("--keep_wav", action="store_true", help="Also keep the intermediate .wav file")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    sf2 = find_soundfont(args.soundfont)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    midis = collect_inputs(args.inputs, args.recursive)
    print(f"Soundfont: {sf2}")
    print(f"Converting {len(midis)} file(s)...")
    for i, midi in enumerate(midis, 1):
        mp3 = convert_one(midi, out_dir, sf2, args.sample_rate, args.gain, args.bitrate, args.keep_wav)
        print(f"  [{i}/{len(midis)}] {midi} -> {mp3}")
    print("Done.")

if __name__ == "__main__":
    main()
