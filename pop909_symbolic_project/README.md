# Assignment 2 POP909 Symbolic Generation Project

This project implements two symbolic generation tasks:

1. **Symbolic unconditioned generation**: train a melody language model on POP909 `MELODY` tracks and sample a new melody from scratch. Output: `symbolic_unconditioned.mid`.
2. **Symbolic conditioned generation / harmonization**: given a melody, predict a chord sequence and render a piano accompaniment. Output: `symbolic_conditioned.mid`.

The fun presentation hook is: same melody, different harmonizers — melody alone, always-common-chord baseline, pop-loop baseline, learned Transformer harmonizer with block chords, arpeggios, and ballad broken-chords.

## Get POP909

```bash
git clone https://github.com/music-x-lab/POP909-Dataset.git
```

The path you pass should contain folders like `001/001.mid`, `001/beat_midi.txt`, and `001/chord_midi.txt`:

```bash
python train_all.py --pop909_root ../POP909-Dataset/POP909
```

## Install

```bash
pip install -r requirements.txt
```

## Smoke test without POP909

```bash
python make_toy_pop909.py --out_dir toy_POP909 --n_songs 40
python train_all.py --pop909_root toy_POP909 --epochs_melody 3 --epochs_chord 3 --max_songs 40 --out_dir outputs_toy
```

## Real run

Start small:

```bash
python train_all.py --pop909_root ../POP909-Dataset/POP909 --max_songs 120 --epochs_melody 10 --epochs_chord 10 --out_dir outputs
```

Then use all songs / more epochs if time allows:

```bash
python train_all.py --pop909_root ../POP909-Dataset/POP909 --max_songs 909 --epochs_melody 25 --epochs_chord 25 --out_dir outputs
```

The code writes the required files both inside `outputs/` and in the project root:

```text
symbolic_unconditioned.mid
symbolic_conditioned.mid
```

## Output files for presentation

```text
outputs/symbolic_unconditioned.mid
outputs/melody_prompt_only.mid
outputs/baseline_always_common_chord.mid
outputs/baseline_pop_loop.mid
outputs/symbolic_conditioned_block.mid
outputs/symbolic_conditioned_arpeggio.mid
outputs/symbolic_conditioned_ballad.mid
outputs/eda_summary.csv
outputs/eval_summary.csv
outputs/pitch_histogram.png
outputs/chord_histogram_top30.png
outputs/melody_training_curve.png
outputs/chord_training_curve.png
```

## Notebook / presentation outline

Task 1:
- Data: POP909 melody tracks.
- Preprocessing: highest active melody note per half-beat, or REST.
- Model: LSTM next-token language model.
- Baseline: unigram pitch/rest model.
- Metrics: validation perplexity, pitch range, rest ratio, subjective listening.

Task 2:
- Data: POP909 melody track + `chord_midi.txt`.
- Preprocessing: melody pitch at beat midpoint + bar position -> chord label.
- Model: Transformer encoder sequence tagger.
- Baselines: most-common chord and simple pop loop.
- Metrics: chord accuracy, top-3 accuracy, melody-note-in-chord rate, chord-change rate, subjective listening.

## Update: less boring harmonization

If `symbolic_conditioned_block.mid` sounds like the same left hand forever, use the updated guided decoder rather than raw argmax chord prediction. The current default is already guided:

```bash
python train_all.py \
  --pop909_root ../POP909-Dataset/POP909 \
  --max_songs 200 \
  --epochs_melody 10 \
  --epochs_chord 15 \
  --out_dir outputs \
  --decode_strategy guided \
  --chord_temperature 0.9 \
  --chord_top_k 10 \
  --max_same_chord 3
```

The important new files are:

- `outputs/debug_raw_argmax_chords.mid`: what the classifier alone predicts. This may be boring and is useful for comparison.
- `outputs/symbolic_conditioned_block.mid`: guided chord sequence with voice-led block voicings.
- `outputs/symbolic_conditioned_arpeggio.mid`: same guided chords, broken into arpeggios.
- `outputs/symbolic_conditioned_ballad.mid`: same guided chords, pop-ballad accompaniment pattern.
- `outputs/symbolic_conditioned_syncopated.mid`: same guided chords, a more rhythmic accompaniment.

The guided decoder combines model probabilities with melody-note compatibility, learned POP909 chord-transition priors, a phrase-level chord-change bonus, and a penalty for repeating the same chord too many times. This is intentionally more musical than pure argmax decoding.

Useful knobs:

- More variety: increase `--chord_temperature 1.1`, increase `--decode_phrase_change_bonus 0.5`, or lower `--max_same_chord 2`.
- More conservative / model-faithful: lower `--chord_temperature 0.7`, increase `--decode_model_weight 1.5`, or raise `--max_same_chord 4`.
- More melody-fitting: increase `--decode_fit_weight 1.6`.
- More POP909-like progressions: increase `--decode_transition_weight 0.8`.

For presentation, play `debug_raw_argmax_chords.mid` first, then `symbolic_conditioned_ballad.mid`. The contrast demonstrates why generation is not just classifier accuracy: decoding and rendering choices matter musically.
