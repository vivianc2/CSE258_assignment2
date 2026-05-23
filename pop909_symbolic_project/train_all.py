from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from pop909_symbolic.midi_io import list_song_dirs, load_song
from pop909_symbolic.datasets import prepare_song, split_songs, build_chord_vocab, melody_unigram_perplexity, majority_chord_accuracy
from pop909_symbolic.training import train_melody_model, train_chord_model, train_remi_lm, make_outputs
from pop909_symbolic.analysis import make_eda_tables, plot_history

def parse_args():
    p=argparse.ArgumentParser(description='POP909 symbolic melody generation + harmonization')
    p.add_argument('--pop909_root', required=True); p.add_argument('--out_dir', default='outputs'); p.add_argument('--max_songs', type=int, default=200); p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--frame_step', type=float, default=.5); p.add_argument('--seq_len', type=int, default=64); p.add_argument('--stride', type=int, default=16); p.add_argument('--max_beats', type=int, default=256); p.add_argument('--max_chords', type=int, default=64)
    p.add_argument('--epochs_melody', type=int, default=15); p.add_argument('--epochs_chord', type=int, default=15); p.add_argument('--batch_size', type=int, default=64); p.add_argument('--chord_batch_size', type=int, default=16); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--melody_emb', type=int, default=96); p.add_argument('--melody_hidden', type=int, default=192); p.add_argument('--melody_layers', type=int, default=2); p.add_argument('--chord_dim', type=int, default=128); p.add_argument('--chord_heads', type=int, default=4); p.add_argument('--chord_layers', type=int, default=3)
    p.add_argument('--temperature', type=float, default=.9); p.add_argument('--top_k', type=int, default=12); p.add_argument('--gen_melody_frames', type=int, default=192); p.add_argument('--gen_chord_beats', type=int, default=96); p.add_argument('--tempo', type=int, default=105)
    # Harmonization decoding controls. Use guided by default; argmax is included only for debugging.
    p.add_argument('--decode_strategy', choices=['guided','argmax'], default='guided')
    p.add_argument('--chord_top_k', type=int, default=10)
    p.add_argument('--chord_beam_size', type=int, default=6)
    p.add_argument('--chord_temperature', type=float, default=0.9)
    p.add_argument('--decode_model_weight', type=float, default=1.0)
    p.add_argument('--decode_fit_weight', type=float, default=1.2)
    p.add_argument('--decode_transition_weight', type=float, default=0.55)
    p.add_argument('--decode_repeat_penalty', type=float, default=0.42)
    p.add_argument('--decode_phrase_change_bonus', type=float, default=0.32)
    p.add_argument('--max_same_chord', type=int, default=3)
    # REMI Transformer melody LM (Task 1 primary).
    p.add_argument('--train_remi', action='store_true', default=True, help='Train the REMI Transformer LM (Task 1 primary model).')
    p.add_argument('--no_remi', dest='train_remi', action='store_false', help='Skip the REMI Transformer (LSTM baseline only).')
    p.add_argument('--epochs_remi', type=int, default=80)
    p.add_argument('--remi_seq_len', type=int, default=384)
    p.add_argument('--remi_stride', type=int, default=192)
    p.add_argument('--remi_batch_size', type=int, default=32)
    p.add_argument('--remi_lr', type=float, default=3e-4)
    p.add_argument('--remi_d_model', type=int, default=384)
    p.add_argument('--remi_nhead', type=int, default=6)
    p.add_argument('--remi_layers', type=int, default=6)
    p.add_argument('--remi_ff', type=int, default=1536)
    p.add_argument('--remi_dropout', type=float, default=0.1)
    p.add_argument('--remi_gen_bars', type=int, default=32)
    p.add_argument('--remi_temperature', type=float, default=1.0)
    p.add_argument('--remi_top_k', type=int, default=0)
    p.add_argument('--remi_top_p', type=float, default=0.92)
    p.add_argument('--remi_repeat_penalty', type=float, default=1.15)
    return p.parse_args()

def main():
    args=parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dirs=list_song_dirs(args.pop909_root); dirs=dirs[:args.max_songs] if args.max_songs else dirs; raw=[]; failed=[]
    for d in dirs:
        try:
            s=load_song(d)
            if s: raw.append(s)
            else: failed.append(d.name)
        except Exception as e: failed.append(f'{d.name}: {e}')
    if len(raw) < 10: raise RuntimeError(f'Only loaded {len(raw)} valid songs. Check --pop909_root or run make_toy_pop909.py.')
    songs=[prepare_song(s, step=args.frame_step) for s in raw]; train,val,test=split_songs(songs,args.seed)
    print(f'Prepared songs: train={len(train)}, val={len(val)}, test={len(test)}, failed={len(failed)}')
    make_eda_tables(songs,out); chord_to_idx, idx_to_chord=build_chord_vocab(train,args.max_chords); (out/'chord_vocab.json').write_text(json.dumps({'idx_to_chord':idx_to_chord,'chord_to_idx':chord_to_idx}, indent=2))
    base_ppl=melody_unigram_perplexity(train,val); base_acc=majority_chord_accuracy(train,val); print(f'Baseline melody unigram perplexity: {base_ppl:.3f}'); print(f'Baseline majority chord accuracy: {base_acc:.3f}')
    melody_model, mhist, _=train_melody_model(train,val,out,args); plot_history(mhist,out/'melody_training_curve.png','LSTM melody LM (baseline)')
    chord_model, chist, _=train_chord_model(train,val,chord_to_idx,idx_to_chord,out,args); plot_history(chist,out/'chord_training_curve.png','Chord Transformer training')
    remi_model = None; rhist = []
    if getattr(args, 'train_remi', True):
        remi_model, rhist, _ = train_remi_lm(train, val, out, args)
        plot_history(rhist, out/'remi_training_curve.png', 'REMI Transformer melody LM')
    info=make_outputs(melody_model,chord_model,train,val,test,chord_to_idx,idx_to_chord,out,args,remi_model=remi_model)
    fm=mhist[-1] if mhist else {}; fc=chist[-1] if chist else {}; fr=rhist[-1] if rhist else {}
    rows=[{'metric':'melody_unigram_baseline_ppl','value':base_ppl},{'metric':'melody_lstm_val_ppl','value':fm.get('val_ppl',float('nan'))},{'metric':'melody_remi_transformer_val_ppl','value':fr.get('val_ppl',float('nan'))},{'metric':'chord_majority_baseline_acc','value':base_acc},{'metric':'chord_transformer_val_acc','value':fc.get('val_acc',float('nan'))},{'metric':'chord_transformer_val_top3_acc','value':fc.get('val_top3_acc',float('nan'))},{'metric':'melody_note_in_pred_chord_rate','value':fc.get('melody_note_in_pred_chord_rate',float('nan'))},{'metric':'pred_chord_change_rate','value':fc.get('pred_chord_change_rate',float('nan'))}]
    pd.DataFrame(rows).to_csv(out/'eval_summary.csv',index=False); (out/'run_summary.json').write_text(json.dumps({'args':vars(args),'n_loaded_songs':len(raw),'n_failed':len(failed),'failed_preview':failed[:10],'chord_vocab_size':len(idx_to_chord),'outputs':info,'final_melody':fm,'final_chord':fc,'final_remi':fr},indent=2))
    print('\nDone. Required files copied to current directory: symbolic_unconditioned.mid and symbolic_conditioned.mid')
    print('Fun extras are in', out)
if __name__ == '__main__': main()
