from __future__ import annotations
 
import math
from typing import Optional
 
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# TUPE Multi-Head Attention
# ---------------------------------------------------------------------------
class TUPEMultiHeadAttention(nn.Module):
    """
    Transformer with Untied Positional Encoding (Ke et al., 2020).
 
    Score d'attention = content-to-content + position-to-position (découplés).
 
        α_ij = (1/√2d) · (x_i Wq)(x_j Wk)ᵀ  +  (1/√2d) · (p_i Uq)(p_j Uk)ᵀ
 
    - x_i : embedding de contenu (cell embedding)
    - p_i : embedding positionnel (projection du timestamp)
    - Wq, Wk : projections contenu
    - Uq, Uk : projections positionnelles (indépendantes)
 
    Pas d'addition x+p avant l'attention — les deux flux sont strictement séparés.
    """
 
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
 
        # Projections contenu (Q, K, V)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
 
        # Projections positionnelles indépendantes (Uq, Uk)
        self.Uq = nn.Linear(d_model, d_model, bias=False)
        self.Uk = nn.Linear(d_model, d_model, bias=False)
 
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.scale    = 1.0 / math.sqrt(2 * self.d_k)   # 1/√(2 d_k) comme dans le papier
 
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) → (B, n_heads, L, d_k)"""
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
 
    def forward(
        self,
        content: torch.Tensor,          # (B, L, d_model)
        pos_emb: torch.Tensor,           # (B, L, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, L) bool, True = ignore
        attn_mask: Optional[torch.Tensor] = None,          # (L, L) causal mask
    ) -> torch.Tensor:
        # --- Contenu ---
        Q_c = self._split_heads(self.Wq(content))   # (B, h, L, d_k)
        K_c = self._split_heads(self.Wk(content))
        V   = self._split_heads(self.Wv(content))
 
        # --- Positionnel ---
        Q_p = self._split_heads(self.Uq(pos_emb))
        K_p = self._split_heads(self.Uk(pos_emb))
 
        # Scores TUPE (deux termes additifs)
        scores = (
            self.scale * torch.matmul(Q_c, K_c.transpose(-2, -1)) +
            self.scale * torch.matmul(Q_p, K_p.transpose(-2, -1))
        )  # (B, h, L, L)
 
        # Masque causal (optionnel — non utilisé ici, on ne génère pas en auto-régression)
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
 
        # Masque padding : positions paddées reçoivent -inf avant softmax
        if key_padding_mask is not None:
            # (B, 1, 1, L) broadcast sur (B, h, L, L)
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
 
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
 
        out = torch.matmul(attn_weights, V)              # (B, h, L, d_k)
        out = out.transpose(1, 2).contiguous()           # (B, L, h, d_k)
        out = out.view(out.size(0), out.size(1), -1)     # (B, L, d_model)
        return self.out_proj(out)
 
 
# ---------------------------------------------------------------------------
# Couche Transformer (avec TUPE)
# ---------------------------------------------------------------------------
class TUPETransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = TUPEMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop   = nn.Dropout(dropout)
 
    def forward(
        self,
        content: torch.Tensor,
        pos_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (pré-norm style)
        residual = content
        content  = self.norm1(content)
        attn_out = self.attn(content, pos_emb, key_padding_mask=key_padding_mask)
        content  = residual + self.drop(attn_out)
 
        # Feed-forward
        content = content + self.drop(self.ff(self.norm2(content)))
        return content
 
 
# ---------------------------------------------------------------------------
# Modèle complet
# ---------------------------------------------------------------------------
class MobilityTransformer(nn.Module):
    """
    Transformer pour prédiction de cellule mobile avec TUPE.
 
    Paramètres
    ----------
    n_cells      : nombre de cellules réelles (EOS = n_cells + 1, PAD = 0)
    d_model      : dimension des embeddings (128 par défaut)
    n_heads      : têtes d'attention (4 par défaut)
    n_layers     : couches transformer (2 par défaut)
    d_ff         : dimension FFN interne (256 par défaut)
    max_seq_len  : longueur maximale acceptée (512)
    dropout      : taux de dropout
    """
 
    def __init__(
        self,
        n_cells: int,
        d_model: int   = 128,
        n_heads: int   = 4,
        n_layers: int  = 2,
        d_ff: int      = 256,
        max_seq_len: int = 512,
        pad_cell_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_cells    = n_cells
        self.d_model    = d_model
        self.eos_id     = n_cells + 1       # indice EOS
        vocab_size      = n_cells + 2       # 0=PAD, 1..N=cellules, N+1=EOS
 
        self.pad_cell_id = pad_cell_id
 
        # ── Embedding de contenu (cellule) ──────────────────────────────────
        # padding_idx=0 → le gradient sur PAD reste nul, son embedding = 0
        self.cell_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=self.pad_cell_id)
 
        # ── Projection temporelle pour TUPE ─────────────────────────────────
        # Le timestamp normalisé [0,1] est projeté vers d_model via un petit MLP
        # (sinusoïdal learnable) pour donner p_i.
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
 
        # ── Couches transformer ─────────────────────────────────────────────
        self.layers = nn.ModuleList([
            TUPETransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
 
        # ── Tête de prédiction ───────────────────────────────────────────────
        # Prédit parmi toutes les cellules + EOS (pas PAD)
        self.head = nn.Linear(d_model, vocab_size)
 
        self._init_weights()
 
    def _init_weights(self):
        nn.init.normal_(self.cell_emb.weight, std=0.02)
        # Le vecteur padding reste à zéro
        with torch.no_grad():
            self.cell_emb.weight[self.pad_cell_id].zero_()
 
    def forward(
        self,
        cells: torch.Tensor,          # (B, L) LongTensor
        times: torch.Tensor,          # (B, L) FloatTensor normalisé [0,1]
        padding_mask: torch.Tensor,   # (B, L) BoolTensor — True = pad
    ) -> torch.Tensor:
        """
        Retourne les logits (B, L, vocab_size).
        Les positions paddées sont présentes dans la sortie mais ignorées
        lors du calcul de la loss (via ignore_index).
        """
        # Contenu
        content = self.cell_emb(cells)                    # (B, L, d_model)
 
        # Positionnel TUPE : projection indépendante des timestamps
        pos_emb = self.time_proj(times.unsqueeze(-1))     # (B, L, d_model)
 
        # Passage dans les couches
        for layer in self.layers:
            content = layer(content, pos_emb, key_padding_mask=padding_mask)
 
        content = self.final_norm(content)
        return self.head(content)                         # (B, L, vocab_size)