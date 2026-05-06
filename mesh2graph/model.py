from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xgboost as xgb
except Exception:
    xgb = None


class FaceEncoder(nn.Module):
    def __init__(self, face_in_dim: int, hidden_dim: int, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(face_in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_face: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.in_proj(x_face)))
        h = h.unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        return h.squeeze(0)


class GraphFaceToTopologyNet(nn.Module):
    def __init__(
        self,
        face_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_spaces: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_spaces = int(max_spaces)

        self.encoder = FaceEncoder(face_in_dim, hidden_dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)

        self.ff_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.space_queries = nn.Parameter(torch.randn(self.max_spaces, hidden_dim) * 0.02)
        self.global_to_space = nn.Linear(hidden_dim, hidden_dim)
        self.space_norm = nn.LayerNorm(hidden_dim)

        self.s_exist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.space_proj = nn.Linear(hidden_dim, hidden_dim)
        self.face_proj_for_sf = nn.Linear(hidden_dim, hidden_dim)
        self.sf_bias = nn.Parameter(torch.zeros(1))

    def encode(self, x_face: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_face)

    def ff_logits(self, h_face: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
        src = pair_idx[0]
        dst = pair_idx[1]
        z = torch.cat([h_face[src], h_face[dst]], dim=-1)
        return self.ff_head(z).squeeze(-1)

    def build_space_latents(self, h_face: torch.Tensor) -> torch.Tensor:
        if h_face.size(0) == 0:
            pooled = torch.zeros((1, self.hidden_dim), device=h_face.device, dtype=h_face.dtype)
        else:
            pooled = h_face.mean(dim=0, keepdim=True)
        s = self.space_queries + self.global_to_space(pooled)
        return self.space_norm(s)

    def s_exist_logits(self, s_latent: torch.Tensor) -> torch.Tensor:
        return self.s_exist_head(s_latent).squeeze(-1)

    def sf_logits(self, h_face: torch.Tensor, s_latent: torch.Tensor) -> torch.Tensor:
        s = self.space_proj(s_latent)
        f = self.face_proj_for_sf(h_face)
        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        return torch.matmul(s, f.transpose(0, 1)) * scale + self.sf_bias

    def predict_all(self, x_face: torch.Tensor):
        h = self.encode(x_face)
        s_latent = self.build_space_latents(h)
        s_logits = self.s_exist_logits(s_latent)
        sf_logits = self.sf_logits(h, s_latent)
        return h, s_logits, sf_logits


class DeepSetsFaceEncoder(nn.Module):
    def __init__(self, face_in_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = face_in_dim
        for _ in range(max(1, int(num_layers))):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.phi = nn.Sequential(*layers)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_face: torch.Tensor) -> torch.Tensor:
        h_local = self.phi(x_face)
        if h_local.size(0) == 0:
            return h_local
        h_global = self.rho(h_local.mean(dim=0, keepdim=True))
        return self.out_norm(h_local + h_global)


class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(y))
        y = self.ffn(x)
        return self.norm2(x + self.drop(y))


class _InducedSetAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_inducing: int = 16, dropout: float = 0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(num_inducing, hidden_dim) * 0.02)
        self.attn_induce = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_decode = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_i = nn.LayerNorm(hidden_dim)
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        i = self.inducing.unsqueeze(0).expand(x.size(0), -1, -1)
        h_i, _ = self.attn_induce(i, x, x, need_weights=False)
        h_i = self.norm_i(i + self.drop(h_i))
        h_x, _ = self.attn_decode(x, h_i, h_i, need_weights=False)
        x = self.norm_x(x + self.drop(h_x))
        y = self.ffn(x)
        return self.norm_out(x + self.drop(y))


class SetTransformerFaceEncoder(nn.Module):
    def __init__(
        self,
        face_in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        num_inducing: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(face_in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.isab_layers = nn.ModuleList(
            [_InducedSetAttentionBlock(hidden_dim, num_heads, num_inducing=num_inducing, dropout=dropout) for _ in range(max(1, int(num_layers)))]
        )
        self.sab = _SelfAttentionBlock(hidden_dim, num_heads, dropout=dropout)

    def forward(self, x_face: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.in_proj(x_face))).unsqueeze(0)
        for layer in self.isab_layers:
            x = layer(x)
        x = self.sab(x)
        return x.squeeze(0)


class PerceiverFaceEncoder(nn.Module):
    def __init__(
        self,
        face_in_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        num_latents: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(face_in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) * 0.02)
        self.cross_in = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_out = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.latent_blocks = nn.ModuleList([_SelfAttentionBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(max(1, int(num_layers)))])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_face: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.in_proj(x_face))).unsqueeze(0)
        if x.size(1) == 0:
            return x.squeeze(0)

        lat = self.latents.unsqueeze(0).expand(1, -1, -1)
        lat_in, _ = self.cross_in(lat, x, x, need_weights=False)
        lat = lat + self.drop(lat_in)
        for block in self.latent_blocks:
            lat = block(lat)

        x_out, _ = self.cross_out(x, lat, lat, need_weights=False)
        x = self.norm(x + self.drop(x_out))
        return x.squeeze(0)


class DeepSetsFaceToTopologyNet(nn.Module):
    def __init__(
        self,
        face_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_spaces: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_spaces = int(max_spaces)

        self.encoder = DeepSetsFaceEncoder(face_in_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        self.ff_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.space_queries = nn.Parameter(torch.randn(self.max_spaces, hidden_dim) * 0.02)
        self.global_to_space = nn.Linear(hidden_dim, hidden_dim)
        self.space_norm = nn.LayerNorm(hidden_dim)

        self.s_exist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.space_proj = nn.Linear(hidden_dim, hidden_dim)
        self.face_proj_for_sf = nn.Linear(hidden_dim, hidden_dim)
        self.sf_bias = nn.Parameter(torch.zeros(1))

    def encode(self, x_face: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_face)

    def ff_logits(self, h_face: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
        src = pair_idx[0]
        dst = pair_idx[1]
        z = torch.cat([h_face[src], h_face[dst]], dim=-1)
        return self.ff_head(z).squeeze(-1)

    def build_space_latents(self, h_face: torch.Tensor) -> torch.Tensor:
        if h_face.size(0) == 0:
            pooled = torch.zeros((1, self.hidden_dim), device=h_face.device, dtype=h_face.dtype)
        else:
            pooled = h_face.mean(dim=0, keepdim=True)
        s = self.space_queries + self.global_to_space(pooled)
        return self.space_norm(s)

    def s_exist_logits(self, s_latent: torch.Tensor) -> torch.Tensor:
        return self.s_exist_head(s_latent).squeeze(-1)

    def sf_logits(self, h_face: torch.Tensor, s_latent: torch.Tensor) -> torch.Tensor:
        s = self.space_proj(s_latent)
        f = self.face_proj_for_sf(h_face)
        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        return torch.matmul(s, f.transpose(0, 1)) * scale + self.sf_bias

    def predict_all(self, x_face: torch.Tensor):
        h = self.encode(x_face)
        s_latent = self.build_space_latents(h)
        s_logits = self.s_exist_logits(s_latent)
        sf_logits = self.sf_logits(h, s_latent)
        return h, s_logits, sf_logits


class SetTransformerFaceToTopologyNet(DeepSetsFaceToTopologyNet):
    def __init__(
        self,
        face_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_inducing: int = 16,
        dropout: float = 0.1,
        max_spaces: int = 64,
    ):
        super().__init__(
            face_in_dim=face_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_spaces=max_spaces,
        )
        self.encoder = SetTransformerFaceEncoder(
            face_in_dim=face_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_inducing=num_inducing,
            dropout=dropout,
        )


class PerceiverFaceToTopologyNet(DeepSetsFaceToTopologyNet):
    def __init__(
        self,
        face_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_latents: int = 32,
        dropout: float = 0.1,
        max_spaces: int = 64,
    ):
        super().__init__(
            face_in_dim=face_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_spaces=max_spaces,
        )
        self.encoder = PerceiverFaceEncoder(
            face_in_dim=face_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_latents=num_latents,
            dropout=dropout,
        )


class MLPFaceEncoder(nn.Module):
    def __init__(self, face_in_dim: int, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = face_in_dim
        for _ in range(max(1, int(num_layers))):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_face: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x_face)
        return self.norm(h)


class MLPFaceToTopologyNet(DeepSetsFaceToTopologyNet):
    def __init__(
        self,
        face_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_spaces: int = 64,
    ):
        super().__init__(
            face_in_dim=face_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_spaces=max_spaces,
        )
        self.encoder = MLPFaceEncoder(
            face_in_dim=face_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )


class XGBoostFFBaseline:
    """Classical FF-edge baseline using hand-crafted pair features and XGBoost.

    This baseline is offline and not used by the PyTorch end-to-end training loop.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        if params is None:
            params = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
            }
        self.params = dict(params)
        self.model = None

    def _check_backend(self):
        if xgb is None:
            raise ImportError("xgboost is not installed. Install with: pip install xgboost")

    @staticmethod
    def build_pair_features(x_face: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
        src = pair_idx[0].astype(np.int64)
        dst = pair_idx[1].astype(np.int64)
        a = x_face[src]
        b = x_face[dst]
        return np.concatenate([a, b, np.abs(a - b), a * b], axis=1)

    def fit(self, x_face: np.ndarray, pair_idx: np.ndarray, labels: np.ndarray):
        self._check_backend()
        features = self.build_pair_features(x_face, pair_idx)
        clf = xgb.XGBClassifier(**self.params)
        clf.fit(features, labels.astype(np.int64))
        self.model = clf
        return self

    def predict_proba(self, x_face: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoostFFBaseline must be fitted before predict_proba.")
        features = self.build_pair_features(x_face, pair_idx)
        proba = self.model.predict_proba(features)
        return proba[:, 1]


MODEL_REGISTRY = {
    "TopoTransformer": GraphFaceToTopologyNet,
    "DeepSets": DeepSetsFaceToTopologyNet,
    "MLP": MLPFaceToTopologyNet,
    "Perceiver": PerceiverFaceToTopologyNet,
    "SetTransformer": SetTransformerFaceToTopologyNet,
    "XGBoost": XGBoostFFBaseline,
}


def build_model(model_name: str, face_in_dim: int, hidden_dim: int, max_spaces: int):
    model_cls = MODEL_REGISTRY[model_name]
    if model_name == "XGBoost":
        return model_cls()
    return model_cls(face_in_dim=face_in_dim, hidden_dim=hidden_dim, max_spaces=max_spaces)
