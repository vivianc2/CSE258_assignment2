# 20-minute presentation structure

1. Opening: Learning pop melody and harmonization from POP909.
2. Dataset / EDA: POP909 has MELODY, BRIDGE, PIANO tracks and beat/chord annotations; show pitch and chord histograms.
3. Task 1: unconditioned melody generation. Half-beat melody tokens -> LSTM LM -> `symbolic_unconditioned.mid`.
4. Task 2: conditioned harmonization. Beat-level melody + bar position -> Transformer chord labels -> rendered piano accompaniment.
5. Evaluation: perplexity for melody; chord accuracy/top-3; melody-note-in-chord rate; explain why subjective listening matters.
6. Fun demo sequence: melody only, always-common chord, pop loop, learned block chords, learned arpeggio, learned ballad.
7. Limitations: chord labels are not unique ground truth; renderer is rule-based; future work could generate full piano accompaniment.
