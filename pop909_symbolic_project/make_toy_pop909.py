from __future__ import annotations
import argparse, random
from pathlib import Path
import mido
CHORDS=['C:maj','G:maj','A:min','F:maj']; CHORD_PITCHES={'C:maj':[48,52,55],'G:maj':[43,47,50],'A:min':[45,48,52],'F:maj':[41,45,48]}; SCALE=[60,62,64,65,67,69,71,72]
def append_abs(track, events):
    events.sort(key=lambda x:(x[0],0 if x[1].type=='note_off' else 1)); last=0
    for tick,msg in events: msg.time=max(0,tick-last); track.append(msg); last=tick
def write_song(root, idx, bars=16):
    sid=f'{idx:03d}'; d=root/sid; d.mkdir(parents=True, exist_ok=True); mid=mido.MidiFile(ticks_per_beat=480); mel=mido.MidiTrack(); pia=mido.MidiTrack(); mid.tracks += [mel,pia]; mel.append(mido.MetaMessage('track_name', name='MELODY', time=0)); mel.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(105), time=0)); pia.append(mido.MetaMessage('track_name', name='PIANO', time=0)); me=[]; pe=[]; rng=random.Random(idx); beats=bars*4
    for b in range(beats):
        chord=CHORDS[(b//4+idx)%len(CHORDS)]; pcs={p%12 for p in CHORD_PITCHES[chord]}; opts=[p for p in SCALE if p%12 in pcs] or SCALE; pitch=rng.choice(opts if rng.random()<.75 else SCALE); st=b*480
        if rng.random()>.12: me += [(st,mido.Message('note_on',note=pitch,velocity=86,time=0,channel=0)),(st+(360 if rng.random()<.8 else 240),mido.Message('note_off',note=pitch,velocity=0,time=0,channel=0))]
        for n in CHORD_PITCHES[chord]: pe += [(st,mido.Message('note_on',note=n,velocity=55,time=0,channel=1)),(st+420,mido.Message('note_off',note=n,velocity=0,time=0,channel=1))]
    append_abs(mel,me); append_abs(pia,pe); mid.save(d/f'{sid}.mid')
    (d/'beat_midi.txt').write_text(' '.join(str(x) for b in range(beats) for x in [float(b),1.0,1.0 if b%4==0 else 0.0]))
    (d/'chord_midi.txt').write_text(' '.join(str(x) for bar in range(bars) for x in [float(bar*4),float((bar+1)*4),CHORDS[(bar+idx)%len(CHORDS)]]))
    (d/'key_audio.txt').write_text(f'0.0 {float(beats)} C:maj')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out_dir',default='toy_POP909'); ap.add_argument('--n_songs',type=int,default=40); args=ap.parse_args(); root=Path(args.out_dir); root.mkdir(exist_ok=True)
    for i in range(1,args.n_songs+1): write_song(root,i)
    print(f'Wrote {args.n_songs} toy songs to {root}')
if __name__=='__main__': main()
