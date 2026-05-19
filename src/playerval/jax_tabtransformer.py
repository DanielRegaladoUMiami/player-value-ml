"""TabTransformer hand-rolled in pure JAX.

Each input row becomes a sequence of column-tokens:
  - Categorical column c with value v -> embedding[c][v] in R^d
  - Numerical column j with value x  -> learned linear projection of x to R^d

So a row of (n_cat + n_num) columns becomes an (n_cat + n_num, d) token matrix.
We then run L layers of multi-head self-attention OVER THE COLUMNS (not time),
pool, and project to a scalar — log_ratio.

This mirrors the TabTransformer paper but is written from first principles
in JAX: no Flax, no Equinox. The point is the craft, not a SOTA result.

For evaluation: trained with Optax AdamW + MSE loss on y_log_ratio, scored
with MAE in log-ratio space (matching everything else in this project).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class TTConfig:
    n_num: int                # number of numerical columns
    n_cat: int                # number of categorical columns
    cat_cardinalities: tuple[int, ...]  # vocab size per categorical col
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


def _truncated_normal(key, shape, std: float = 0.02):
    return jax.random.truncated_normal(key, -2.0, 2.0, shape) * std


def init_params(cfg: TTConfig, key) -> dict[str, Any]:
    keys = jax.random.split(key, 3 + cfg.n_layers * 6 + cfg.n_cat)
    it = iter(keys)

    params: dict[str, Any] = {}

    # Per-column projections for numerical features: each gets W (1, d_model) + b
    params["num_proj_W"] = _truncated_normal(next(it), (cfg.n_num, cfg.d_model))
    params["num_proj_b"] = jnp.zeros((cfg.n_num, cfg.d_model))

    # Per-column embeddings for categorical features
    params["cat_emb"] = [
        _truncated_normal(next(it), (card, cfg.d_model))
        for card in cfg.cat_cardinalities
    ]

    # Per-column positional embedding so the model can tell columns apart
    n_cols = cfg.n_num + cfg.n_cat
    params["col_pos"] = _truncated_normal(next(it), (n_cols, cfg.d_model))

    # Transformer layers
    layers = []
    for _ in range(cfg.n_layers):
        layer = {
            "attn_norm": jnp.ones((cfg.d_model,)),
            "wq": _truncated_normal(next(it), (cfg.d_model, cfg.d_model)),
            "wk": _truncated_normal(next(it), (cfg.d_model, cfg.d_model)),
            "wv": _truncated_normal(next(it), (cfg.d_model, cfg.d_model)),
            "wo": _truncated_normal(next(it), (cfg.d_model, cfg.d_model)),
            "mlp_norm": jnp.ones((cfg.d_model,)),
            "w_ff_up":   _truncated_normal(next(it), (cfg.d_model, cfg.d_ff)),
            "w_ff_down": _truncated_normal(next(it), (cfg.d_ff, cfg.d_model)),
        }
        layers.append(layer)
    params["layers"] = layers

    # Final head: pool over columns -> MLP -> scalar
    params["head_norm"] = jnp.ones((cfg.d_model,))
    params["head_W"] = _truncated_normal(next(it), (cfg.d_model, 1))
    params["head_b"] = jnp.zeros((1,))
    return params


def rms_norm(x, w, eps: float = 1e-5):
    var = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
    inv = lax.rsqrt(var + eps).astype(x.dtype)
    return x * inv * w


def attention(x, layer, cfg: TTConfig):
    """Multi-head self-attention OVER THE COLUMN SEQUENCE.

    x: (B, T, D) where T = n_cols. No causal mask (columns are unordered).
    """
    B, T, D = x.shape
    H, dh = cfg.n_heads, cfg.d_head
    q = (x @ layer["wq"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    k = (x @ layer["wk"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    v = (x @ layer["wv"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    scale = 1.0 / jnp.sqrt(jnp.float32(dh))
    scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
    attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    out = jnp.einsum("bhij,bhjd->bhid", attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, H * dh)
    return out @ layer["wo"]


def block(x, layer, cfg: TTConfig):
    h = rms_norm(x, layer["attn_norm"])
    x = x + attention(h, layer, cfg)
    h = rms_norm(x, layer["mlp_norm"])
    ff = jax.nn.gelu(h @ layer["w_ff_up"]) @ layer["w_ff_down"]
    return x + ff


def forward(params, x_num, x_cat, cfg: TTConfig) -> jnp.ndarray:
    """Forward pass returning (B,) predicted log_ratio.

    x_num: (B, n_num) float
    x_cat: (B, n_cat) int (category indices)
    """
    B = x_num.shape[0]

    # Numerical: project each column independently with its own weight vector
    # num_proj_W shape (n_num, d_model). Each scalar x_num[b, j] -> W[j] * x + b[j]
    num_tokens = x_num[:, :, None] * params["num_proj_W"][None, :, :] + params["num_proj_b"][None, :, :]
    # (B, n_num, d_model)

    # Categorical: lookup each column's embedding
    cat_tokens_list = []
    for j in range(cfg.n_cat):
        emb = params["cat_emb"][j]                     # (card_j, d_model)
        cat_tokens_list.append(emb[x_cat[:, j]])       # (B, d_model)
    if cat_tokens_list:
        cat_tokens = jnp.stack(cat_tokens_list, axis=1)  # (B, n_cat, d_model)
        tokens = jnp.concatenate([num_tokens, cat_tokens], axis=1)
    else:
        tokens = num_tokens

    # Add per-column positional embedding
    tokens = tokens + params["col_pos"][None, :, :]

    # Transformer layers
    for layer in params["layers"]:
        tokens = block(tokens, layer, cfg)

    # Pool: mean across columns
    pooled = jnp.mean(tokens, axis=1)             # (B, d_model)
    pooled = rms_norm(pooled, params["head_norm"])
    out = (pooled @ params["head_W"]).squeeze(-1) + params["head_b"][0]
    return out


def count_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(jnp.size(x) for x in leaves))
