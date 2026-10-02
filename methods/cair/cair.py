"""CAIR interpolator, residual refiner, and ensemble inference."""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.cair.config import (
    BEST_ENSEMBLE_MEMBERS,
    BEST_INFERENCE_CONFIG,
    BEST_MEMBER_KWARGS,
)
from methods.cair.backbone import CAIRBackbone, CGM_STD

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DEFAULT_CKPT_DIR = os.path.join(_ROOT, "method_checkpoints", "cair")
DEFAULT_STRIDE = int(BEST_INFERENCE_CONFIG["stride"])
DEFAULT_N_REFINEMENTS = int(BEST_INFERENCE_CONFIG["n_refinements"])


class _Interp(nn.Module):
    """Pluggable fully-neural interpolator -> y0 (B,T,1). Head zero-init so a
    residual model starts at the transformer output and learns the base curve."""

    def __init__(self, in_ch: int, hidden: int, layers: int, kind: str = "gru"):
        super().__init__()
        self.kind = kind
        if kind in ("gru", "lstm"):
            rnn = nn.GRU if kind == "gru" else nn.LSTM
            self.net = rnn(
                in_ch,
                hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1 if layers > 1 else 0.0,
            )
            self.head = nn.Linear(2 * hidden, 1)
        elif kind == "tcn":
            self.proj = nn.Conv1d(in_ch, hidden, 1)
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d)
                    for d in (1, 2, 4, 8, 16, 32, 64)
                ]
            )
            self.head = nn.Conv1d(hidden, 1, 1)
        elif kind == "attn":
            self.inp = nn.Linear(in_ch, hidden)
            enc = nn.TransformerEncoderLayer(
                hidden,
                4,
                hidden * 2,
                0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.net = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(hidden, 1)
        else:
            raise ValueError(kind)
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, f):  # f (B,T,in_ch)
        if self.kind in ("gru", "lstm"):
            h, _ = self.net(f)
            return self.head(h)
        if self.kind == "tcn":
            x = self.proj(f.transpose(1, 2))
            for c in self.convs:
                x = x + F.gelu(c(x))
            return self.head(x).transpose(1, 2)
        return self.head(self.net(self.inp(f)))  # attn


class CAIR(CAIRBackbone):
    """Fully-neural interpolate-and-refine model (single member of the ensemble)."""

    def __init__(
        self,
        interp_hidden: int = int(BEST_MEMBER_KWARGS["interp_hidden"]),
        interp_layers: int = int(BEST_MEMBER_KWARGS["interp_layers"]),
        inject: str = str(BEST_MEMBER_KWARGS["inject"]),
        interp_type: str = str(BEST_MEMBER_KWARGS["interp_type"]),
        interp_in: str = str(BEST_MEMBER_KWARGS["interp_in"]),
        **kw,
    ):
        super().__init__(**kw)
        assert inject in ("value", "value_slope", "residual")
        assert interp_in in ("basic", "rich")
        self.inject = inject
        self.interp_type = interp_type
        self.interp_in = interp_in
        in_ch = 2 if interp_in == "basic" else 7
        self.interp = _Interp(in_ch, interp_hidden, interp_layers, interp_type)

    def _interp_feats(self, x_obs, obs_code, pos_idx, tod_idx, cond):
        obs_flag = (obs_code == 1).float().unsqueeze(-1)
        feats = [x_obs * obs_flag, obs_flag]
        if self.interp_in == "rich":
            ang = 2 * np.pi * tod_idx.float().unsqueeze(-1) / 288.0
            posn = (pos_idx.float() / float(self.window)).unsqueeze(-1)
            feats += [
                torch.sin(ang),
                torch.cos(ang),
                posn,
                cond[..., 19:20],
                cond[..., 20:21],
            ]
        return torch.cat(feats, dim=-1)

    def interp_pred(self, x_obs, obs_code, pos_idx=None, tod_idx=None, cond=None):
        """Stage-1 learned interpolation from the originally-observed points."""
        return self.interp(self._interp_feats(x_obs, obs_code, pos_idx, tod_idx, cond))

    def _encode(
        self, x_obs, obs_code, pos_idx, tod_idx, cond, dom_idx=None, ctx_vec=None
    ):
        y0 = self.interp_pred(x_obs, obs_code, pos_idx, tod_idx, cond)  # (B,T,1)
        y0s = y0.squeeze(-1)
        cond = cond.clone()
        cond[..., 21] = y0s
        cond[..., 23] = y0s  # neural interp -> both value slots
        if self.inject == "value_slope":
            slope = torch.zeros_like(y0s)
            slope[:, 1:] = y0s[:, 1:] - y0s[:, :-1]
            cond[..., 22] = slope
            cond[..., 24] = slope
        else:
            cond[..., 22] = 1.0
            cond[..., 24] = 1.0

        # ctx_vec is forwarded to the base static-context pathway.
        out = super()._encode(x_obs, obs_code, pos_idx, tod_idx, cond, dom_idx, ctx_vec)
        if self.inject == "residual":
            out = out + y0
        return out


def _load_member(ckpt_path: str, device: str) -> CAIR:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck.get("config", {})
    m = CAIR(
        interp_hidden=c.get("hidden", 128),
        interp_layers=c.get("layers", 4),
        inject=c.get("inject", "residual"),
        interp_type=c.get("interp_type", "gru"),
        interp_in=c.get("interp_in", "basic"),
        d_model=c.get("d_model", 128),
        n_layers=c.get("n_layers", 8),
        ff_dim=c.get("ff_dim", 512),
        n_heads=c.get("n_heads", 8),
        window=c.get("window", 576),
        n_datasets=c.get("n_datasets", 1),
        n_extra_cond=c.get("n_extra_cond", 0),
        ctx_dim=c.get("ctx_dim", 0),
    )
    # strict=False so pre-dom_emb published members (no "dom_emb.weight") still load;
    # a missing dom_emb stays zero-init == no domain term == the original behavior.
    info = m.load_state_dict(ck["model"], strict=False)
    missing = [k for k in info.missing_keys if not k.startswith("dom_emb")]
    if missing or info.unexpected_keys:
        raise RuntimeError(
            f"CAIR load mismatch for {ckpt_path}: "
            f"missing={missing} unexpected={list(info.unexpected_keys)}"
        )
    m.eval().to(torch.device(device))
    return m


class CAIRImputer:
    """Fully-neural CAIR imputer (deep ensemble of seed members).

    The prediction is the average of the member ``impute`` outputs. No classical
    interpolation and no length-gate: the result is fully neural.
    """

    def __init__(
        self,
        ckpt_paths: Optional[Sequence[str]] = None,
        ckpt_dir: str = DEFAULT_CKPT_DIR,
        device: str = "cpu",
        stride: int = DEFAULT_STRIDE,
        n_refinements: int = DEFAULT_N_REFINEMENTS,
    ):
        if ckpt_paths is None:
            ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        if not ckpt_paths:
            expected = ", ".join(BEST_ENSEMBLE_MEMBERS)
            raise FileNotFoundError(
                "no CAIR checkpoints found. "
                f"Expected ensemble members like [{expected}] under {ckpt_dir}, "
                "or pass ckpt_paths/ckpt_dir explicitly."
            )
        self.device = device
        self.stride = stride
        self.n_refinements = n_refinements
        self.models: List[CAIR] = [_load_member(p, device) for p in ckpt_paths]
        print(f"CAIR loaded ({len(self.models)}-member ensemble, device {device})")

    def impute(
        self, full_ts: np.ndarray, obs_mask: np.ndarray, modality=None, ctx_vec=None
    ) -> np.ndarray:
        """Fully-neural prediction over the full series (ensemble-averaged).

        modality: optional (T, K) per-timestep modality cond columns.
        ctx_vec:  optional (S,) per-subject static-context vector.
        """
        obs = np.asarray(obs_mask, dtype=np.float32).ravel()
        preds = [
            m.impute(
                full_ts,
                obs,
                device=str(self.device),
                stride=self.stride,
                n_refinements=self.n_refinements,
                modality=modality,
                ctx_vec=ctx_vec,
            )
            for m in self.models
        ]
        return np.mean(preds, axis=0).astype(np.float32)
