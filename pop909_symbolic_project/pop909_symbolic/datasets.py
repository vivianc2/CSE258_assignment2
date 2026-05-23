from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random, numpy as np, torch
from torch.utils.data import Dataset
from .constants import REST, PAD, CHORD_N, CHORD_OOV, CHORD_PAD
from .midi_io import SongData, frame_melody, chord_at_time, pitch_at_time
from .remi import encode_remi, PAD_ID as REMI_PAD_ID, BOS_ID as REMI_BOS_ID, EOS_ID as REMI_EOS_ID

@dataclass
class PreparedSong:
    song_id: str
    melody_frames: List[int]
    beat_melody: List[int]
    beat_positions: List[int]
    chord_labels: List[str]
    remi_tokens: List[int]

def prepare_song(song: SongData, step: float = 0.5) -> PreparedSong:
    total_beats = max(song.beat_times[-1] if song.beat_times else 1.0, max((n.end for n in song.melody_notes), default=1.0))
    frames = frame_melody(song.melody_notes, total_beats, step)
    beat_melody=[]; beat_positions=[]; chord_labels=[]; since_down=0
    beat_dur = float(np.median(np.diff(song.beat_times))) if len(song.beat_times) >= 2 else 1.0
    for i,t in enumerate(song.beat_times[:-1]):
        if i < len(song.downbeats) and song.downbeats[i] == 1: since_down = 0
        mid_t = t + .5 * beat_dur
        beat_melody.append(pitch_at_time(song.melody_notes, mid_t, REST))
        beat_positions.append(since_down)
        chord_labels.append(chord_at_time(song.chord_segments, mid_t, CHORD_N))
        since_down = (since_down + 1) % 8
    remi_tokens = encode_remi(song.melody_notes, song.beat_times, add_bos_eos=True)
    return PreparedSong(song.song_id, frames, beat_melody, beat_positions, chord_labels, remi_tokens)

def split_songs(songs: Sequence[PreparedSong], seed=42, train_frac=.8, val_frac=.1):
    songs=list(songs); random.Random(seed).shuffle(songs); n=len(songs); a=int(n*train_frac); b=int(n*val_frac)
    return songs[:a], songs[a:a+b], songs[a+b:]

class MelodyWindowDataset(Dataset):
    def __init__(self, songs, seq_len=64, stride=16):
        self.examples=[]
        for s in songs:
            seq=s.melody_frames
            for st in range(0, max(0,len(seq)-seq_len-1), stride):
                self.examples.append(torch.tensor(seq[st:st+seq_len+1], dtype=torch.long))
        if not self.examples: raise ValueError('No melody windows; reduce seq_len or use more songs.')
    def __len__(self): return len(self.examples)
    def __getitem__(self,i):
        a=self.examples[i]; return a[:-1], a[1:]

class RemiWindowDataset(Dataset):
    """Pack each song's REMI token sequence into fixed-length training windows.

    Long songs are sliced with overlap (stride < seq_len) so the model sees the
    interior of every song multiple times. Windows shorter than seq_len+1 are
    right-padded with PAD so they're still usable for short songs.
    """
    def __init__(self, songs, seq_len: int = 384, stride: int = 192):
        self.seq_len = seq_len
        self.examples: List[torch.Tensor] = []
        for s in songs:
            seq = list(s.remi_tokens)
            if len(seq) < 4:
                continue
            if len(seq) <= seq_len + 1:
                pad = [REMI_PAD_ID] * (seq_len + 1 - len(seq))
                self.examples.append(torch.tensor(seq + pad, dtype=torch.long))
                continue
            for st in range(0, len(seq) - seq_len, stride):
                self.examples.append(torch.tensor(seq[st:st + seq_len + 1], dtype=torch.long))
            # Always include the tail so the EOS region gets seen.
            tail = seq[-(seq_len + 1):]
            self.examples.append(torch.tensor(tail, dtype=torch.long))
        if not self.examples:
            raise ValueError("No REMI windows; check song lengths.")
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        a = self.examples[i]
        return a[:-1], a[1:]


class ChordSequenceDataset(Dataset):
    def __init__(self, songs, chord_to_idx: Dict[str,int], max_len=256):
        self.items=[]; self.chord_to_idx=chord_to_idx
        for s in songs:
            n=min(len(s.beat_melody),len(s.chord_labels),max_len)
            if n < 8: continue
            mel=torch.tensor(s.beat_melody[:n], dtype=torch.long)
            pos=torch.tensor([p%8 for p in s.beat_positions[:n]], dtype=torch.long)
            ch=torch.tensor([chord_to_idx.get(c, chord_to_idx[CHORD_OOV]) for c in s.chord_labels[:n]], dtype=torch.long)
            self.items.append((s.song_id, mel, pos, ch))
        if not self.items: raise ValueError('No chord examples.')
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def collate_chord_batch(batch):
    ids, ms, ps, cs = zip(*batch); L=max(len(x) for x in ms); B=len(batch)
    mel=torch.full((B,L), PAD, dtype=torch.long); pos=torch.zeros((B,L), dtype=torch.long); ch=torch.full((B,L), CHORD_PAD, dtype=torch.long); mask=torch.zeros((B,L), dtype=torch.bool)
    for i,(m,p,c) in enumerate(zip(ms,ps,cs)):
        n=len(m); mel[i,:n]=m; pos[i,:n]=p; ch[i,:n]=c; mask[i,:n]=True
    return ids, mel, pos, ch, mask

def build_chord_vocab(train_songs, max_chords=64) -> Tuple[Dict[str,int], List[str]]:
    cnt=Counter()
    for s in train_songs: cnt.update(s.chord_labels)
    labels=[CHORD_N, CHORD_OOV]
    for c,_ in cnt.most_common(max_chords):
        if c not in labels: labels.append(c)
    return {c:i for i,c in enumerate(labels)}, labels

def melody_unigram_perplexity(train_songs, val_songs) -> float:
    cnt=Counter(); total=0; vocab=130
    for s in train_songs: cnt.update(s.melody_frames); total += len(s.melody_frames)
    nll=0.; n=0
    for s in val_songs:
        for x in s.melody_frames:
            nll -= np.log((cnt[x]+1)/(total+vocab)); n += 1
    return float(np.exp(nll/max(1,n)))

def majority_chord_accuracy(train_songs, val_songs) -> float:
    cnt=Counter()
    for s in train_songs: cnt.update(s.chord_labels)
    maj=cnt.most_common(1)[0][0] if cnt else CHORD_N; ok=tot=0
    for s in val_songs:
        for c in s.chord_labels: ok += int(c==maj); tot += 1
    return ok/max(1,tot)
