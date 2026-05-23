from __future__ import annotations
from pathlib import Path
from collections import Counter
import math, shutil, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .constants import PAD, CHORD_PAD, REST
from .datasets import MelodyWindowDataset, ChordSequenceDataset, RemiWindowDataset, collate_chord_batch
from .models import MelodyLSTM, MelodyTransformerLM, ChordTransformer
from .midi_io import write_midi, write_midi_from_notes, melody_note_in_chord_rate
from .harmonize import build_transition_logprobs, guided_decode_chords, summarize_chord_sequence
from . import remi as RR


def train_melody_model(train_songs, val_songs, out_dir: Path, args):
    train_ds=MelodyWindowDataset(train_songs, args.seq_len, args.stride); val_ds=MelodyWindowDataset(val_songs, args.seq_len, args.seq_len)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True); val_loader=DataLoader(val_ds,batch_size=args.batch_size)
    device=torch.device(args.device); model=MelodyLSTM(emb_dim=args.melody_emb, hidden_dim=args.melody_hidden, num_layers=args.melody_layers).to(device); opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    hist=[]; best=float('inf'); path=out_dir/'melody_lstm.pt'
    for ep in range(1,args.epochs_melody+1):
        model.train(); total=tok=0
        for x,y in tqdm(train_loader, desc=f'Melody epoch {ep}', leave=False):
            x=x.to(device); y=y.to(device); logits=model(x); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)), y.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total += loss.item()*y.numel(); tok += y.numel()
        val=evaluate_melody_loss(model,val_loader,device); row={'epoch':ep,'train_loss':total/max(1,tok),'val_loss':val,'val_ppl':math.exp(min(20,val))}; hist.append(row)
        if val < best: best=val; torch.save({'model_state':model.state_dict(),'args':vars(args)}, path)
    model.load_state_dict(torch.load(path, map_location=device)['model_state']); return model,hist,path

@torch.no_grad()
def evaluate_melody_loss(model, loader, device):
    model.eval(); total=tok=0
    for x,y in loader:
        x=x.to(device); y=y.to(device); logits=model(x); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)), y.reshape(-1), ignore_index=PAD, reduction='sum'); total += loss.item(); tok += y.numel()
    return total/max(1,tok)

@torch.no_grad()
def sample_melody(model, length=160, temperature=1.0, top_k=12, seed_token=REST, device='cpu'):
    model.eval(); seq=[seed_token]
    for _ in range(length-1):
        x=torch.tensor([seq[-128:]], dtype=torch.long, device=device); logits=model(x)[0,-1] / max(temperature,1e-6); logits[PAD] = -1e9
        if top_k and top_k>0:
            vals,idx=torch.topk(logits, k=min(top_k, logits.numel())); probs=torch.softmax(vals,dim=-1); nxt=idx[torch.multinomial(probs,1).item()].item()
        else:
            nxt=torch.multinomial(torch.softmax(logits,dim=-1),1).item()
        seq.append(int(nxt))
    return seq


# ---------------------------------------------------------------------------
# REMI Transformer melody language model
# ---------------------------------------------------------------------------

def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return max(1e-3, step / max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_remi_lm(train_songs, val_songs, out_dir: Path, args):
    """Train a decoder-only Transformer LM on REMI tokens. Returns (model, history, ckpt_path)."""
    seq_len = getattr(args, "remi_seq_len", 384)
    stride = getattr(args, "remi_stride", seq_len // 2)
    train_ds = RemiWindowDataset(train_songs, seq_len=seq_len, stride=stride)
    val_ds = RemiWindowDataset(val_songs, seq_len=seq_len, stride=seq_len)
    train_loader = DataLoader(train_ds, batch_size=getattr(args, "remi_batch_size", 32), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=getattr(args, "remi_batch_size", 32))
    device = torch.device(args.device)
    model = MelodyTransformerLM(
        d_model=getattr(args, "remi_d_model", 384),
        nhead=getattr(args, "remi_nhead", 6),
        num_layers=getattr(args, "remi_layers", 6),
        dim_feedforward=getattr(args, "remi_ff", 1536),
        dropout=getattr(args, "remi_dropout", 0.1),
        max_len=max(seq_len, 1024),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"REMI Transformer parameters: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=getattr(args, "remi_lr", 3e-4),
                            weight_decay=0.01, betas=(0.9, 0.95))
    epochs = getattr(args, "epochs_remi", args.epochs_melody)
    total_steps = max(1, len(train_loader) * epochs)
    warmup = max(50, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _cosine_warmup(s, warmup, total_steps))
    hist = []; best = float("inf"); path = out_dir / "remi_melody_transformer.pt"; step = 0
    pad_id = RR.PAD_ID
    for ep in range(1, epochs + 1):
        model.train(); total = tok = 0
        for x, y in tqdm(train_loader, desc=f"REMI epoch {ep}", leave=False):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            mask = (y != pad_id)
            total += loss.item() * mask.sum().item(); tok += mask.sum().item()
        val = _evaluate_remi_loss(model, val_loader, device, pad_id)
        row = {"epoch": ep, "train_loss": total / max(1, tok), "val_loss": val,
               "val_ppl": math.exp(min(20, val)), "lr": opt.param_groups[0]["lr"]}
        hist.append(row)
        if val < best:
            best = val
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "vocab_size": RR.VOCAB_SIZE}, path)
    model.load_state_dict(torch.load(path, map_location=device)["model_state"])
    return model, hist, path


@torch.no_grad()
def _evaluate_remi_loss(model, loader, device, pad_id):
    model.eval(); total = tok = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        mask = (y != pad_id)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=pad_id, reduction="sum")
        total += loss.item(); tok += mask.sum().item()
    return total / max(1, tok)


def _remi_allowed_mask(prev_token: int, device) -> torch.Tensor:
    """Boolean mask of token ids that are syntactically valid after `prev_token`.

    REMI is a strict grammar: a Position must be followed by a Pitch, a Pitch
    must be followed by a Duration. Letting the model emit malformed sequences
    leads to dropped notes; we enforce the grammar at sample time.

    Two extra rules beyond pure grammar prevent degenerate outputs from a
    not-fully-converged model:
      - after BAR we forbid another BAR (force at least one note per bar);
      - we never let the model emit EOS itself — the sampling loop injects EOS
        once enough bars have been generated.
    """
    mask = torch.zeros(RR.VOCAB_SIZE, dtype=torch.bool, device=device)
    if prev_token == RR.BOS_ID:
        mask[RR.BAR_ID] = True
    elif prev_token == RR.BAR_ID:
        mask[RR.POS_BASE:RR.PITCH_BASE] = True
    elif RR.POS_BASE <= prev_token < RR.PITCH_BASE:
        mask[RR.PITCH_BASE:RR.DUR_BASE] = True
    elif RR.PITCH_BASE <= prev_token < RR.DUR_BASE:
        mask[RR.DUR_BASE:RR.VOCAB_SIZE] = True
    elif RR.DUR_BASE <= prev_token < RR.VOCAB_SIZE:
        mask[RR.BAR_ID] = True
        mask[RR.POS_BASE:RR.PITCH_BASE] = True
    else:
        mask[RR.BAR_ID] = True
    # Never emit PAD / BOS / EOS during sampling — the loop injects EOS.
    mask[RR.PAD_ID] = False
    mask[RR.BOS_ID] = False
    mask[RR.EOS_ID] = False
    return mask


@torch.no_grad()
def sample_remi(model: MelodyTransformerLM, n_bars: int = 32, temperature: float = 1.0,
                top_k: int = 0, top_p: float = 0.92, pitch_repeat_penalty: float = 1.15,
                device: str = "cpu", seed: int | None = None,
                max_tokens: int = 4096) -> list:
    """Sample a REMI token sequence with nucleus + grammar constraints + repetition penalty.

    Stops when `n_bars` Bar tokens have been emitted or EOS appears.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = [RR.BOS_ID, RR.BAR_ID]
    bars_seen = 1
    # Track recent pitch ids for repetition penalty.
    recent_pitches: list[int] = []
    for _ in range(max_tokens):
        ctx = torch.tensor([tokens[-model.max_len:]], dtype=torch.long, device=device)
        logits = model(ctx)[0, -1].clone()
        # Repetition penalty on recently-emitted pitches: scale their logits down.
        if recent_pitches and pitch_repeat_penalty != 1.0:
            for pid in set(recent_pitches[-8:]):
                if logits[pid] > 0: logits[pid] /= pitch_repeat_penalty
                else: logits[pid] *= pitch_repeat_penalty
        # Grammar mask.
        allowed = _remi_allowed_mask(tokens[-1], logits.device)
        logits = logits.masked_fill(~allowed, float("-inf"))
        logits = logits / max(temperature, 1e-6)
        # Top-k.
        if top_k and top_k > 0:
            vals, _ = torch.topk(logits, k=min(top_k, logits.numel()))
            cutoff = vals[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        # Top-p (nucleus).
        if top_p and 0 < top_p < 1.0:
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=-1)
            keep = cum <= top_p
            keep[0] = True
            keep_mask = torch.zeros_like(probs, dtype=torch.bool)
            keep_mask[sorted_idx[keep]] = True
            probs = probs * keep_mask
            s = probs.sum()
            if not torch.isfinite(s) or s <= 0:
                # Fall back to the masked argmax.
                nxt = int(torch.argmax(probs))
            else:
                probs = probs / s
                nxt = int(torch.multinomial(probs, 1).item())
        else:
            s = probs.sum()
            if not torch.isfinite(s) or s <= 0:
                nxt = int(torch.argmax(probs))
            else:
                nxt = int(torch.multinomial(probs / s, 1).item())
        tokens.append(nxt)
        if RR.PITCH_BASE <= nxt < RR.DUR_BASE:
            recent_pitches.append(nxt)
        if nxt == RR.BAR_ID:
            bars_seen += 1
            if bars_seen > n_bars:
                tokens.append(RR.EOS_ID); break
    return tokens


def _chord_class_weights(train_songs, chord_to_idx, device):
    counts=torch.ones(len(chord_to_idx), dtype=torch.float32)
    for song in train_songs:
        for c in song.chord_labels:
            if c in chord_to_idx:
                counts[chord_to_idx[c]] += 1.0
    weights = counts.sum() / (counts * len(chord_to_idx))
    weights = torch.clamp(weights, 0.25, 4.0)
    return weights.to(device)

def train_chord_model(train_songs, val_songs, chord_to_idx, idx_to_chord, out_dir: Path, args):
    train_ds=ChordSequenceDataset(train_songs,chord_to_idx,args.max_beats); val_ds=ChordSequenceDataset(val_songs,chord_to_idx,args.max_beats)
    train_loader=DataLoader(train_ds,batch_size=args.chord_batch_size,shuffle=True,collate_fn=collate_chord_batch); val_loader=DataLoader(val_ds,batch_size=args.chord_batch_size,collate_fn=collate_chord_batch)
    device=torch.device(args.device); model=ChordTransformer(len(idx_to_chord), d_model=args.chord_dim, nhead=args.chord_heads, num_layers=args.chord_layers, max_len=args.max_beats).to(device); opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); class_weights=_chord_class_weights(train_songs, chord_to_idx, device)
    hist=[]; best=-1; path=out_dir/'chord_transformer.pt'
    for ep in range(1,args.epochs_chord+1):
        model.train(); total=tok=0
        for _,mel,pos,ch,mask in tqdm(train_loader, desc=f'Chord epoch {ep}', leave=False):
            mel=mel.to(device); pos=pos.to(device); ch=ch.to(device); mask=mask.to(device); logits=model(mel,pos,key_padding_mask=~mask); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)), ch.reshape(-1), ignore_index=CHORD_PAD, weight=class_weights)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total += loss.item()*mask.sum().item(); tok += mask.sum().item()
        metrics=evaluate_chord_model(model,val_loader,device,idx_to_chord); row={'epoch':ep,'train_loss':total/max(1,tok),**metrics}; hist.append(row)
        if metrics['val_acc'] > best: best=metrics['val_acc']; torch.save({'model_state':model.state_dict(),'args':vars(args),'idx_to_chord':idx_to_chord,'chord_to_idx':chord_to_idx}, path)
    model.load_state_dict(torch.load(path, map_location=device)['model_state']); return model,hist,path

@torch.no_grad()
def evaluate_chord_model(model, loader, device, idx_to_chord):
    model.eval(); correct=correct3=total=0; loss_sum=0.; mp=[]; pc=[]; changes=den=0
    for _,mel,pos,ch,mask in loader:
        mel=mel.to(device); pos=pos.to(device); ch=ch.to(device); mask=mask.to(device); logits=model(mel,pos,key_padding_mask=~mask); active=mask & (ch != CHORD_PAD); preds=logits.argmax(-1); top3=logits.topk(k=min(3,logits.size(-1)), dim=-1).indices
        loss_sum += F.cross_entropy(logits.reshape(-1,logits.size(-1)), ch.reshape(-1), ignore_index=CHORD_PAD, reduction='sum').item(); correct += ((preds==ch)&active).sum().item(); correct3 += ((top3==ch.unsqueeze(-1)).any(-1)&active).sum().item(); total += active.sum().item()
        for b in range(mel.shape[0]):
            last=None
            for t in range(mel.shape[1]):
                if active[b,t]:
                    c=idx_to_chord[int(preds[b,t])]; mp.append(int(mel[b,t])); pc.append(c)
                    if last is not None: changes += int(c != last); den += 1
                    last=c
    return {'val_loss':loss_sum/max(1,total),'val_acc':correct/max(1,total),'val_top3_acc':correct3/max(1,total),'melody_note_in_pred_chord_rate':melody_note_in_chord_rate(mp,pc),'pred_chord_change_rate':changes/max(1,den)}

@torch.no_grad()
def predict_chords(model, prepared_song, chord_to_idx, idx_to_chord, device, max_beats=160, transition_logprobs=None, args=None):
    model.eval()
    n=min(max_beats,len(prepared_song.beat_melody))
    mel=torch.tensor([prepared_song.beat_melody[:n]], dtype=torch.long, device=device)
    pos=torch.tensor([[p%8 for p in prepared_song.beat_positions[:n]]], dtype=torch.long, device=device)
    mask=mel!=PAD
    logits=model(mel,pos,key_padding_mask=~mask)

    # Raw argmax is useful for debugging, but often collapses to a boring common chord.
    raw_preds=logits.argmax(-1)[0].cpu().tolist()
    raw_chords=[idx_to_chord[i] for i in raw_preds]

    if args is not None and getattr(args, 'decode_strategy', 'guided') == 'argmax':
        chords=raw_chords
    else:
        chords=guided_decode_chords(
            logits[0],
            prepared_song.beat_melody[:n],
            idx_to_chord,
            transition_logprobs=transition_logprobs,
            top_k=getattr(args, 'chord_top_k', 10),
            beam_size=getattr(args, 'chord_beam_size', 6),
            temperature=getattr(args, 'chord_temperature', 0.9),
            model_weight=getattr(args, 'decode_model_weight', 1.0),
            fit_weight=getattr(args, 'decode_fit_weight', 1.2),
            transition_weight=getattr(args, 'decode_transition_weight', 0.55),
            repeat_penalty=getattr(args, 'decode_repeat_penalty', 0.42),
            phrase_change_bonus=getattr(args, 'decode_phrase_change_bonus', 0.32),
            max_same=getattr(args, 'max_same_chord', 3),
            seed=getattr(args, 'seed', 42),
        )
    return chords, prepared_song.beat_melody[:n], raw_chords

def make_outputs(melody_model, chord_model, train_songs, val_songs, test_songs, chord_to_idx, idx_to_chord, out_dir: Path, args, remi_model=None):
    # Baseline (LSTM frame-level LM): kept for the presentation A/B story.
    gen=sample_melody(melody_model, length=args.gen_melody_frames, temperature=args.temperature, top_k=args.top_k, seed_token=REST, device=args.device)
    write_midi(out_dir/'baseline_lstm_melody.mid', melody_frames=gen, step=args.frame_step, tempo=args.tempo)

    # Primary submission: REMI Transformer melody LM with nucleus + grammar-constrained sampling.
    if remi_model is not None:
        remi_tokens = sample_remi(
            remi_model,
            n_bars=getattr(args, 'remi_gen_bars', 32),
            temperature=getattr(args, 'remi_temperature', 1.0),
            top_k=getattr(args, 'remi_top_k', 0),
            top_p=getattr(args, 'remi_top_p', 0.92),
            pitch_repeat_penalty=getattr(args, 'remi_repeat_penalty', 1.15),
            device=args.device,
            seed=getattr(args, 'seed', 42),
        )
        decoded = RR.decode_remi(remi_tokens)
        notes = RR.decoded_notes_to_midi_notes(decoded, tempo_bpm=args.tempo)
        write_midi_from_notes(out_dir/'symbolic_unconditioned.mid', notes, tempo=args.tempo)
        # Also save the raw token sequence for the notebook / appendix.
        (out_dir/'symbolic_unconditioned_remi_tokens.txt').write_text(
            ' '.join(str(t) for t in remi_tokens))
    else:
        write_midi(out_dir/'symbolic_unconditioned.mid', melody_frames=gen, step=args.frame_step, tempo=args.tempo)

    song=test_songs[0] if test_songs else val_songs[0]
    transition_logprobs = build_transition_logprobs(train_songs, chord_to_idx)
    chords, beat_melody, raw_chords = predict_chords(
        chord_model, song, chord_to_idx, idx_to_chord, args.device, args.gen_chord_beats,
        transition_logprobs=transition_logprobs, args=args
    )
    frames=[]
    for p in beat_melody: frames.extend([p,p])

    for style in ['block','arpeggio','ballad','syncopated']:
        write_midi(out_dir/f'symbolic_conditioned_{style}.mid', melody_frames=frames, chord_labels=chords, step=args.frame_step, tempo=args.tempo, style=style)
    shutil.copyfile(out_dir/'symbolic_conditioned_ballad.mid', out_dir/'symbolic_conditioned.mid')

    # Diagnostic: what the neural classifier alone wanted, before the musical decoder.
    write_midi(out_dir/'debug_raw_argmax_chords.mid', melody_frames=frames, chord_labels=raw_chords, step=args.frame_step, tempo=args.tempo, style='block')

    common=[most_common_non_oov_chord(train_songs)]*len(chords)
    loop=pop_loop_like_chords(train_songs,len(chords))
    write_midi(out_dir/'baseline_always_common_chord.mid', melody_frames=frames, chord_labels=common, step=args.frame_step, tempo=args.tempo, style='block')
    write_midi(out_dir/'baseline_pop_loop.mid', melody_frames=frames, chord_labels=loop, step=args.frame_step, tempo=args.tempo, style='ballad')
    write_midi(out_dir/'melody_prompt_only.mid', melody_frames=frames, step=args.frame_step, tempo=args.tempo)
    shutil.copyfile(out_dir/'symbolic_unconditioned.mid', Path('symbolic_unconditioned.mid'))
    shutil.copyfile(out_dir/'symbolic_conditioned.mid', Path('symbolic_conditioned.mid'))
    return {
        'prompt_song_id':song.song_id,
        'raw_argmax_chords': summarize_chord_sequence(raw_chords),
        'guided_chords': summarize_chord_sequence(chords),
    }

def most_common_non_oov_chord(train_songs):
    cnt=Counter()
    for s in train_songs: cnt.update([c for c in s.chord_labels if c not in ('N','OOV')])
    return cnt.most_common(1)[0][0] if cnt else 'C:maj'

def pop_loop_like_chords(train_songs,n):
    candidates=['C:maj','G:maj','A:min','F:maj']; avail={c for s in train_songs for c in s.chord_labels}
    if not all(c in avail for c in candidates):
        cnt=Counter(); [cnt.update([c for c in s.chord_labels if c not in ('N','OOV')]) for s in train_songs]; candidates=[c for c,_ in cnt.most_common(4)] or ['C:maj']
    return [candidates[i%len(candidates)] for i in range(n)]
