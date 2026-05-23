from __future__ import annotations
import math, torch
import torch.nn as nn
from .constants import PITCH_VOCAB, PAD
from .remi import VOCAB_SIZE as REMI_VOCAB_SIZE, PAD_ID as REMI_PAD_ID

class MelodyLSTM(nn.Module):
    def __init__(self, vocab_size=PITCH_VOCAB, emb_dim=96, hidden_dim=192, num_layers=2, dropout=.2):
        super().__init__(); self.emb=nn.Embedding(vocab_size, emb_dim, padding_idx=PAD); self.lstm=nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers>1 else 0, batch_first=True); self.out=nn.Linear(hidden_dim, vocab_size)
    def forward(self,x): return self.out(self.lstm(self.emb(x))[0])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__(); pe=torch.zeros(max_len,d_model); pos=torch.arange(max_len,dtype=torch.float32).unsqueeze(1); div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:pe[:,1::2].shape[1]]); self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self,x): return x + self.pe[:,:x.shape[1],:]

class MelodyTransformerLM(nn.Module):
    """Decoder-only Transformer language model over REMI tokens.

    Standard GPT-style stack: token + learned absolute position embedding,
    pre-norm Transformer blocks, causal self-attention via the encoder layer
    + a square mask, weight-tied output head. Small by GPT standards (~3M
    params at the defaults below) but large enough to dramatically outperform
    the LSTM on long-range structure in REMI sequences.
    """
    def __init__(self, vocab_size: int = REMI_VOCAB_SIZE, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1,
                 max_len: int = 1024, pad_id: int = REMI_PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.register_buffer("_causal_cache_len", torch.tensor(0), persistent=False)
        self._causal_mask: torch.Tensor | None = None

    def _get_causal_mask(self, T: int, device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.size(0) < T or self._causal_mask.device != device:
            mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask = mask
        return self._causal_mask[:T, :T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        if T > self.max_len:
            x = x[:, -self.max_len:]; T = self.max_len
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(positions)
        attn_mask = self._get_causal_mask(T, x.device)
        pad_mask = (x == self.pad_id)
        h = self.blocks(h, mask=attn_mask, src_key_padding_mask=pad_mask)
        return self.head(self.ln_f(h))


class ChordTransformer(nn.Module):
    def __init__(self, n_chords, pitch_vocab=PITCH_VOCAB, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=.15, max_len=512):
        super().__init__(); self.pitch_emb=nn.Embedding(pitch_vocab,d_model,padding_idx=PAD); self.pos_emb=nn.Embedding(8,d_model); self.pe=PositionalEncoding(d_model,max_len); layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True,activation='gelu'); self.encoder=nn.TransformerEncoder(layer,num_layers=num_layers); self.out=nn.Linear(d_model,n_chords)
    def forward(self, melody_tokens, bar_positions, key_padding_mask=None):
        x=self.pe(self.pitch_emb(melody_tokens)+self.pos_emb(bar_positions.clamp(0,7))); return self.out(self.encoder(x, src_key_padding_mask=key_padding_mask))
