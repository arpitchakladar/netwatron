"""
models/state_pooling.py
-------------------------
Stage D (Sec. 17-19) and Stage F (Sec. 23): compress a variable-sized,
masked set of flow embeddings H_t = {h~_t,1 ... h~_t,Nt} into a
network-state z_t in R^128 (or Z_t in R^{M x 128} for R5).

All variants respect the padding mask: padded flows must never influence
attention weights or pooled means (Sec. 14).

R4 (graph) is intentionally NOT implemented as a default path: Sec. 19/77
item 7 explicitly forbid auto-enabling a graph encoder when
source/destination IP is unavailable. GraphStateEncoder below raises if
constructed without `has_ip_columns=True` being explicitly confirmed by the
caller, to make this guard hard to silently bypass.
"""

import torch
import torch.nn as nn

from ..config import StateConfig


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """h: [..., N, d], mask: [..., N] -> [..., d]. R1 (Sec. 17)."""
    mask = mask.unsqueeze(-1)
    summed = (h * mask).sum(dim=-2)
    count = mask.sum(dim=-2).clamp(min=1e-6)
    return summed / count


class AttentionPooling(nn.Module):
    """R2 (Sec. 18): s_i = g(h_i); alpha = softmax(s); z = sum_i alpha_i h_i."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [..., N, d], mask: [..., N]
        scores = self.score(h).squeeze(-1)  # [..., N]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(
            scores, dim=-1
        )  # invalid rows (all -inf) -> handled below
        alpha = torch.nan_to_num(alpha, nan=0.0)
        z = torch.einsum("...n,...nd->...d", alpha, h)
        return z


class SetTransformerPooling(nn.Module):
    """R3 (Sec. 19): self-attention among flows within a window before
    pooling, allowing flow_i <-> flow_j interactions (unlike R1/R2).
    Quadratic in N, so this is not the first implementation choice --
    only enable for smaller max_flows_per_window."""

    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = AttentionPooling(dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        orig_shape = h.shape
        if h.dim() > 3:
            h = h.reshape(-1, orig_shape[-2], orig_shape[-1])
            mask_flat = mask.reshape(-1, orig_shape[-2])
        else:
            mask_flat = mask
        key_padding_mask = mask_flat == 0
        h_enc = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h_enc = torch.nan_to_num(h_enc)  # guards fully-padded rows
        z = self.pool(h_enc, mask_flat)
        if len(orig_shape) > 3:
            z = z.reshape(*orig_shape[:-2], z.shape[-1])
        return z


class MultiTokenPooling(nn.Module):
    """R5 (Sec. 23): M learned latent query tokens attend over the flow set.
    Internally produces Z_t in R^{M x d}, then FLATTENS to R^{M*d} so that
    it satisfies the same "single vector per window" interface (Sec. 24's
    Zhistory in R^{L x 128}) that every downstream module (Dynamics, Risk,
    Stage heads) already assumes. This still lets us test the "is a single
    128-dim bottleneck too restrictive" hypothesis (the effective per-window
    representation is M times wider) without special-casing every other
    module for a ragged extra dimension.

    IMPORTANT: when cfg.state.variant == "R5", set
    cfg.state.state_dim = cfg.flow_encoder.hidden_dim * cfg.state.num_tokens
    so the rest of the pipeline (dynamics hidden sizes, risk/stage heads)
    is constructed with the correct widened dimensionality.
    """

    def __init__(self, dim: int, num_tokens: int, num_heads: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.base_dim = dim
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        orig_shape = h.shape
        if h.dim() > 3:
            h = h.reshape(-1, orig_shape[-2], orig_shape[-1])
            mask_flat = mask.reshape(-1, orig_shape[-2])
        else:
            mask_flat = mask
        B = h.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.attn(
            q, h, h, key_padding_mask=(mask_flat == 0)
        )  # [B, M, d]
        out = out.reshape(
            B, self.num_tokens * self.base_dim
        )  # flatten -> [B, M*d]
        if len(orig_shape) > 3:
            out = out.reshape(*orig_shape[:-2], self.num_tokens * self.base_dim)
        return out


class GraphStateEncoder(nn.Module):
    """R4 (Sec. 20-22): guarded stub. Only meaningful once Src/Dst IP are
    present. See config.DataConfig.has_ip_columns."""

    def __init__(self, dim: int, has_ip_columns: bool):
        super().__init__()
        if not has_ip_columns:
            raise RuntimeError(
                "R4 (graph state) requires source/destination host identity, "
                "which the current CSE-CIC-IDS2018 processed CSV does not "
                "provide (Sec. 19). Refusing to silently fall back to a "
                "graph-less approximation -- set data.has_ip_columns=True "
                "only once real IP/host fields are wired in, per Sec. 77 item 7."
            )
        # Left as a placeholder for a GraphSAGE/GAT/TGN implementation
        # (Sec. 20-21) once host identity is available.
        raise NotImplementedError(
            "Implement once Src/Dst IP fields are available."
        )


def build_state_pooling(
    cfg: StateConfig, has_ip_columns: bool = False
) -> nn.Module:
    if cfg.variant == "R1":

        class _Mean(nn.Module):
            def forward(self, h, mask):
                return masked_mean(h, mask)

        return _Mean()
    if cfg.variant == "R2":
        return AttentionPooling(cfg.state_dim)
    if cfg.variant == "R3":
        return SetTransformerPooling(
            cfg.state_dim, cfg.set_transformer_heads, cfg.set_transformer_layers
        )
    if cfg.variant == "R4":
        return GraphStateEncoder(cfg.state_dim, has_ip_columns)
    if cfg.variant == "R5":
        return MultiTokenPooling(cfg.state_dim, cfg.num_tokens)
    raise ValueError(f"Unknown state pooling variant: {cfg.variant}")
