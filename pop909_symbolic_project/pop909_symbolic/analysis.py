from __future__ import annotations
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from .constants import REST

def make_eda_tables(songs, out_dir: Path):
    rows=[]; pc=Counter(); cc=Counter()
    for s in songs:
        pitches=[p for p in s.melody_frames if p != REST]; pc.update(pitches); cc.update(s.chord_labels)
        rows.append({'song_id':s.song_id,'n_melody_frames':len(s.melody_frames),'n_beats':len(s.beat_melody),'rest_ratio':1-len(pitches)/max(1,len(s.melody_frames)),'lowest_pitch':min(pitches) if pitches else np.nan,'highest_pitch':max(pitches) if pitches else np.nan,'pitch_range':(max(pitches)-min(pitches)) if pitches else np.nan,'unique_pitches':len(set(pitches)),'unique_chords':len(set(s.chord_labels))})
    df=pd.DataFrame(rows); df.to_csv(out_dir/'eda_summary_by_song.csv', index=False); df.describe(include='all').to_csv(out_dir/'eda_summary.csv')
    _plot(pc, out_dir/'pitch_histogram.png', 'Melody pitch histogram', 'MIDI pitch', 'count')
    _plot(Counter(dict(cc.most_common(30))), out_dir/'chord_histogram_top30.png', 'Chord histogram top 30', 'chord', 'count')
    return df

def _plot(counter, path, title, xlabel, ylabel):
    if not counter: return
    items=sorted(counter.items(), key=lambda x: str(x[0])); lab,val=zip(*items); plt.figure(figsize=(10,4)); plt.bar(range(len(val)), val); plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    if len(lab) <= 40: plt.xticks(range(len(lab)), lab, rotation=90)
    plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()

def plot_history(history, path, title):
    if not history: return
    df=pd.DataFrame(history); plt.figure(figsize=(7,4))
    for col in df.columns:
        if col!='epoch' and any(k in col for k in ['loss','acc','ppl']): plt.plot(df['epoch'], df[col], label=col)
    plt.title(title); plt.xlabel('epoch'); plt.legend(); plt.tight_layout(); plt.savefig(path,dpi=180); plt.close(); df.to_csv(path.with_suffix('.csv'), index=False)
