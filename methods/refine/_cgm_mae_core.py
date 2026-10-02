#!/usr/bin/env python3
"""Transformer backbone, conditioning features, and gap curriculum for CAIR."""

from __future__ import annotations
import os, sys, time, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def p(*a):
    return os.path.join(BASE, *a)


# create_physiological_mask is used only by the training Dataset (CGMV66Dataset),
# NOT by inference (impute()). Import it robustly from the clean repo's utils so the
# model + feature code loads anywhere; fall back to None if unavailable.
try:
    from utils.physiological_masking import create_physiological_mask
except Exception:  # pragma: no cover
    try:
        from physiological_masking import create_physiological_mask
    except Exception:
        create_physiological_mask = None

# Refine._encode fills the interpolation slots; conditioning keeps 42 features.
from cgm_datasets import load_dataset_splits

CGM_STD = 42.33
N_HIST = 4
N_COND_BASE = 14
N_COND_WIN = 5
N_COND_GAP = 2
N_COND_LINTERP = 2
N_COND_PCHIP = 2
N_COND_SLOPE = 4
N_COND_ACCEL = 4
N_COND_SMOOTH = 4
N_COND_V60 = (
    N_COND_BASE
    + N_COND_WIN
    + N_COND_GAP
    + N_COND_LINTERP
    + N_COND_PCHIP
    + N_COND_SLOPE
    + N_COND_ACCEL
    + N_COND_SMOOTH
)  # 37
N_COND_CTX = 5  # gap-context: pre_mean, pre_std, post_mean, post_std, gap_length
N_COND = N_COND_V60 + N_COND_CTX  # 42

PASS1_WEIGHT = 0.15
PASS2_WEIGHT = 0.35
PASS3_WEIGHT = 0.50


def compute_gap_context_features(
    eval_mask: np.ndarray, ts: np.ndarray, W: int, N: int = 12
) -> np.ndarray:
    """
    Gap-context features for each masked position (W, 5) float32.

    All positions within the same gap share identical features.
    Observed positions: all zeros (no gap context needed for directly observed values).

    Features per gap:
      col 0: pre_gap_mean  , mean of up to N obs before gap start (CGM level entering gap)
      col 1: pre_gap_std   , std  of up to N obs before gap start (volatility entering gap)
      col 2: post_gap_mean , mean of up to N obs after  gap end   (CGM level exiting gap)
      col 3: post_gap_std  , std  of up to N obs after  gap end   (volatility exiting gap)
      col 4: gap_length    , (gap_end - gap_start) / W            (normalized gap duration)

    Discriminative power:
      Sleep gaps:    long gap_len (0.16+), low pre/post std, stable mean (~-0.45 normalized)
      Meal-post:     short gap_len (0.02-0.05), rising pre_mean, high pre_std
      Ascending:     moderate gap_len, strong positive trend in pre/post means
    """
    obs = eval_mask.astype(bool)
    ts_f = np.asarray(ts, dtype=np.float32)
    out = np.zeros((W, 5), dtype=np.float32)
    obs_indices = np.where(obs)[0].astype(np.int32)

    in_gap = (~obs).astype(np.int8)
    change = np.diff(np.concatenate([[0], in_gap, [0]]))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]

    for gs, ge in zip(starts, ends):
        # Gap-length feature
        out[gs:ge, 4] = float(ge - gs) / W

        # Pre-gap context: last N observed positions before gap start
        left_obs = obs_indices[obs_indices < gs]
        if len(left_obs) >= 1:
            pre_vals = ts_f[left_obs[-min(N, len(left_obs)) :]]
            out[gs:ge, 0] = float(pre_vals.mean())
            out[gs:ge, 1] = float(pre_vals.std()) if len(pre_vals) > 1 else 0.0

        # Post-gap context: first N observed positions after gap end
        right_obs = obs_indices[obs_indices >= ge]
        if len(right_obs) >= 1:
            post_vals = ts_f[right_obs[: min(N, len(right_obs))]]
            out[gs:ge, 2] = float(post_vals.mean())
            out[gs:ge, 3] = float(post_vals.std()) if len(post_vals) > 1 else 0.0

    return out


def precompute_combined(ts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    T = len(ts)
    ts = np.asarray(ts, dtype=np.float32)
    obs = np.asarray(mask, dtype=bool)
    out = np.zeros((T, N_COND_BASE), dtype=np.float32)
    obs_pos = np.where(obs)[0]
    obs_val = ts[obs_pos]
    obs_tod = (obs_pos % 288).astype(np.int32)
    if len(obs_pos):
        counts = np.bincount(obs_tod, minlength=288).astype(np.float64)
        sums = np.bincount(obs_tod, weights=obs_val, minlength=288)
        sum_sq = np.bincount(obs_tod, weights=obs_val**2, minlength=288)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / counts, 0.0)
            vars_ = np.where(
                counts > 1, np.maximum(sum_sq / counts - means**2, 0.0), 0.0
            )
        stds = np.sqrt(vars_).astype(np.float32)
        all_tods = np.arange(T, dtype=np.int32) % 288
        out[:, 0] = means.astype(np.float32)[all_tods]
        out[:, 1] = stds[all_tods]
        tod_starts = np.searchsorted(obs_tod, np.arange(288), side="left")
        tod_ends = np.searchsorted(obs_tod, np.arange(288), side="right")
        for tod in range(288):
            gs, ge = tod_starts[tod], tod_ends[tod]
            if gs == ge:
                continue
            obs_here = obs_pos[gs:ge]
            positions = np.arange(tod, T, 288)
            idxs = np.searchsorted(obs_here, positions, side="left")
            has_prev = idxs > 0
            has_next = idxs < len(obs_here)
            prev_i = np.clip(idxs - 1, 0, len(obs_here) - 1)
            next_i = np.clip(idxs, 0, len(obs_here) - 1)
            out[positions[has_prev], 2] = ts[obs_here[prev_i[has_prev]]]
            out[positions[has_prev], 4] = 1.0
            out[positions[has_next], 3] = ts[obs_here[next_i[has_next]]]
            out[positions[has_next], 5] = 1.0
    for k in range(1, N_HIST + 1):
        for t in range(T):
            s = t - k * 288
            if s >= 0 and obs[s]:
                out[t, 5 + k] = ts[s]
                out[t, 5 + N_HIST + k] = 1.0
    return out


def compute_gap_distances(eval_mask, W):
    obs = eval_mask.astype(bool)
    obs_indices = np.where(obs)[0].astype(np.int32)
    out = np.zeros((W, 2), dtype=np.float32)
    all_pos = np.arange(W, dtype=np.int32)
    if len(obs_indices) == 0:
        out[:, 0] = 1.0
        out[:, 1] = 1.0
        return out
    left_idx = np.searchsorted(obs_indices, all_pos, side="right") - 1
    right_idx = np.searchsorted(obs_indices, all_pos, side="left")
    has_left = left_idx >= 0
    has_right = right_idx < len(obs_indices)
    out[:, 0] = np.where(
        has_left,
        (all_pos - obs_indices[np.clip(left_idx, 0, len(obs_indices) - 1)]) / W,
        1.0,
    ).astype(np.float32)
    out[:, 1] = np.where(
        has_right,
        (obs_indices[np.clip(right_idx, 0, len(obs_indices) - 1)] - all_pos) / W,
        1.0,
    ).astype(np.float32)
    out[obs, :] = 0.0
    return out


def compute_linterp_features(eval_mask, ts, W):
    """REFINE: zero-stub. The linear-interp value-slots (cond cols 21-22) are
    overwritten by the learned neural interpolator in Refine._encode, so no
    classical interpolation is computed here."""
    return np.zeros((W, 2), dtype=np.float32)


def compute_pchip_features(eval_mask, ts, W):
    """REFINE: zero-stub. The PCHIP value-slots (cond cols 23-24) are overwritten
    by the learned neural interpolator in Refine._encode (no scipy / no classical
    interpolation used anywhere in this method)."""
    return np.zeros((W, 2), dtype=np.float32)


def compute_boundary_slope_features(eval_mask, ts, W):
    obs = eval_mask.astype(bool)
    ts_f = np.asarray(ts, dtype=np.float32)
    out = np.zeros((W, 4), dtype=np.float32)
    obs_indices = np.where(obs)[0].astype(np.int32)
    all_pos = np.arange(W, dtype=np.int32)
    if len(obs_indices) < 2:
        return out
    left_idx = np.searchsorted(obs_indices, all_pos, side="right") - 1
    has_left2 = left_idx >= 1
    L_pos = obs_indices[np.clip(left_idx, 0, len(obs_indices) - 1)]
    L2_pos = obs_indices[np.clip(left_idx - 1, 0, len(obs_indices) - 1)]
    L_span = np.maximum(L_pos.astype(np.float32) - L2_pos.astype(np.float32), 1.0)
    out[:, 0] = np.where(has_left2, (ts_f[L_pos] - ts_f[L2_pos]) / L_span, 0.0).astype(
        np.float32
    )
    out[:, 1] = has_left2.astype(np.float32)
    right_idx = np.searchsorted(obs_indices, all_pos, side="left")
    has_right2 = right_idx + 1 < len(obs_indices)
    R_pos = obs_indices[np.clip(right_idx, 0, len(obs_indices) - 1)]
    R2_pos = obs_indices[np.clip(right_idx + 1, 0, len(obs_indices) - 1)]
    R_span = np.maximum(R2_pos.astype(np.float32) - R_pos.astype(np.float32), 1.0)
    out[:, 2] = np.where(has_right2, (ts_f[R2_pos] - ts_f[R_pos]) / R_span, 0.0).astype(
        np.float32
    )
    out[:, 3] = has_right2.astype(np.float32)
    out[obs, :] = 0.0
    return out


def compute_boundary_accel_features(eval_mask, ts, W):
    obs = eval_mask.astype(bool)
    ts_f = np.asarray(ts, dtype=np.float32)
    out = np.zeros((W, 4), dtype=np.float32)
    obs_indices = np.where(obs)[0].astype(np.int32)
    all_pos = np.arange(W, dtype=np.int32)
    if len(obs_indices) < 3:
        return out
    n_obs = len(obs_indices)
    left_idx = np.searchsorted(obs_indices, all_pos, side="right") - 1
    has_left3 = left_idx >= 2
    L1_pos = obs_indices[np.clip(left_idx, 0, n_obs - 1)]
    L2_pos = obs_indices[np.clip(left_idx - 1, 0, n_obs - 1)]
    L3_pos = obs_indices[np.clip(left_idx - 2, 0, n_obs - 1)]
    L1_val = ts_f[L1_pos]
    L2_val = ts_f[L2_pos]
    L3_val = ts_f[L3_pos]
    span12 = np.maximum((L1_pos - L2_pos).astype(np.float32), 1.0)
    span23 = np.maximum((L2_pos - L3_pos).astype(np.float32), 1.0)
    span13 = np.maximum((L1_pos - L3_pos).astype(np.float32), 1.0)
    v1 = (L1_val - L2_val) / span12
    v2 = (L2_val - L3_val) / span23
    out[:, 0] = np.where(has_left3, (v1 - v2) / (span13 / 2.0 + 1e-6), 0.0).astype(
        np.float32
    )
    out[:, 1] = has_left3.astype(np.float32)
    right_idx = np.searchsorted(obs_indices, all_pos, side="left")
    has_right3 = right_idx + 2 < n_obs
    R1_pos = obs_indices[np.clip(right_idx, 0, n_obs - 1)]
    R2_pos = obs_indices[np.clip(right_idx + 1, 0, n_obs - 1)]
    R3_pos = obs_indices[np.clip(right_idx + 2, 0, n_obs - 1)]
    R1_val = ts_f[R1_pos]
    R2_val = ts_f[R2_pos]
    R3_val = ts_f[R3_pos]
    span12r = np.maximum((R2_pos - R1_pos).astype(np.float32), 1.0)
    span23r = np.maximum((R3_pos - R2_pos).astype(np.float32), 1.0)
    span13r = np.maximum((R3_pos - R1_pos).astype(np.float32), 1.0)
    v1r = (R2_val - R1_val) / span12r
    v2r = (R3_val - R2_val) / span23r
    out[:, 2] = np.where(has_right3, (v2r - v1r) / (span13r / 2.0 + 1e-6), 0.0).astype(
        np.float32
    )
    out[:, 3] = has_right3.astype(np.float32)
    out[obs, :] = 0.0
    return out


def compute_smooth_slope_features(eval_mask, ts, W, N=5):
    obs = eval_mask.astype(bool)
    ts_f = np.asarray(ts, dtype=np.float32)
    out = np.zeros((W, 4), dtype=np.float32)
    obs_indices = np.where(obs)[0].astype(np.int32)
    if len(obs_indices) < 2:
        return out

    def ols_slope(t_arr, v_arr):
        if len(t_arr) < 2:
            return 0.0, False
        t_f = t_arr.astype(np.float32)
        v_f = v_arr.astype(np.float32)
        t_mean = t_f.mean()
        v_mean = v_f.mean()
        denom = np.sum((t_f - t_mean) ** 2)
        if denom < 1e-10:
            return 0.0, False
        return float(np.sum((t_f - t_mean) * (v_f - v_mean)) / denom), True

    in_gap = (~obs).astype(np.int8)
    change = np.diff(np.concatenate([[0], in_gap, [0]]))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    for gs, ge in zip(starts, ends):
        left_all = obs_indices[obs_indices < gs]
        if len(left_all) >= 2:
            sl, ok = ols_slope(left_all[-N:], ts_f[left_all[-N:]])
            if ok:
                out[gs:ge, 0] = sl
                out[gs:ge, 1] = 1.0
        right_all = obs_indices[obs_indices >= ge]
        if len(right_all) >= 2:
            sl, ok = ols_slope(right_all[:N], ts_f[right_all[:N]])
            if ok:
                out[gs:ge, 2] = sl
                out[gs:ge, 3] = 1.0
    return out


class CGMMAEV66(nn.Module):
    """Transformer refiner with 42 conditioning features and three training passes."""

    def __init__(
        self,
        window=576,
        d_model=128,
        n_heads=8,
        n_layers=8,
        ff_dim=512,
        dropout=0.1,
        n_datasets=1,
        n_extra_cond=0,
        ctx_dim=0,
    ):
        super().__init__()
        self.window = window
        self.d_model = d_model

        # and an optional per-subject static-context vector (S) -> ctx_proj broadcast.
        self.n_extra_cond = int(n_extra_cond)
        self.ctx_dim = int(ctx_dim)
        self.val_proj = nn.Linear(1, d_model)
        self.obs_emb = nn.Embedding(3, d_model)
        self.day_emb = nn.Embedding(4, d_model)
        self.tod_emb = nn.Embedding(288, d_model)
        self.dom_emb = nn.Embedding(max(1, n_datasets), d_model)
        nn.init.zeros_(
            self.dom_emb.weight
        )  # AI-READI-only training == published REFINE
        self.mask_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cond_proj = nn.Sequential(
            nn.Linear(N_COND + self.n_extra_cond, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)
        # Static-context pathway (mirrors dom_emb): zero-init final layer so it is a
        # no-op at warm-start init -> exact baseline reproduction (parity gate).
        if self.ctx_dim > 0:
            self.ctx_proj = nn.Sequential(
                nn.Linear(self.ctx_dim, 64),
                nn.GELU(),
                nn.Linear(64, d_model),
            )
            nn.init.zeros_(self.ctx_proj[-1].weight)
            nn.init.zeros_(self.ctx_proj[-1].bias)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            ff_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.out_proj = nn.Linear(d_model, 1)

    def _encode(
        self, x_obs, obs_code, pos_idx, tod_idx, cond, dom_idx=None, ctx_vec=None
    ):
        B, T = x_obs.shape[:2]
        val_e = self.val_proj(x_obs)
        obs_e = self.obs_emb(obs_code)
        day_e = self.day_emb(pos_idx // 288)
        tod_e = self.tod_emb(tod_idx)
        cond_e = self.cond_proj(cond)
        ctx_e = day_e + tod_e + obs_e
        if dom_idx is not None:
            # dom_idx: (B,) per-window domain id -> (B,1,d) broadcast over the T axis
            dom_e = self.dom_emb(dom_idx).unsqueeze(1)
            ctx_e = ctx_e + dom_e
        if self.ctx_dim > 0 and ctx_vec is not None:
            # ctx_vec: (B,S) per-subject static context -> (B,1,d) broadcast over T
            ctx_e = ctx_e + self.ctx_proj(ctx_vec).unsqueeze(1)
        is_filled = (obs_code > 0).unsqueeze(-1).expand(B, T, self.d_model)
        mask_exp = self.mask_tok.expand(B, T, -1)
        h = torch.where(is_filled, val_e + ctx_e + cond_e, mask_exp + ctx_e + cond_e)
        h = self.encoder(h)
        return self.out_proj(h)

    def forward(
        self,
        x_obs,
        obs_mask,
        pos_idx,
        tod_idx,
        cond,
        two_pass=False,
        three_pass=False,
        dom_idx=None,
        ctx_vec=None,
    ):
        obs_code1 = obs_mask.long().squeeze(-1)
        pred1 = self._encode(x_obs, obs_code1, pos_idx, tod_idx, cond, dom_idx, ctx_vec)
        if not two_pass and not three_pass:
            return pred1
        obs_flag = obs_mask.bool().squeeze(-1)
        soft_x = torch.where(obs_flag.unsqueeze(-1), x_obs, pred1.detach())
        obs_code2 = torch.where(
            obs_flag,
            torch.ones_like(obs_flag, dtype=torch.long),
            2 * torch.ones_like(obs_flag, dtype=torch.long),
        )
        pred2 = self._encode(
            soft_x, obs_code2, pos_idx, tod_idx, cond, dom_idx, ctx_vec
        )
        if not three_pass:
            return pred1, pred2
        soft_x3 = torch.where(obs_flag.unsqueeze(-1), x_obs, pred2.detach())
        pred3 = self._encode(
            soft_x3, obs_code2, pos_idx, tod_idx, cond, dom_idx, ctx_vec
        )
        return pred1, pred2, pred3

    @torch.no_grad()
    def impute(
        self,
        full_ts,
        obs_mask,
        device="cpu",
        stride=144,
        n_refinements=2,
        modality=None,
        ctx_vec=None,
    ):
        ts = np.asarray(full_ts, dtype=np.float32).ravel()
        mask = np.asarray(obs_mask, dtype=np.float32).ravel()
        T, W = len(ts), self.window
        cond_full = precompute_combined(ts, mask.astype(bool))
        dev = torch.device(device)
        cos_w = (1.0 - np.cos(2 * np.pi * np.arange(W, dtype=np.float32) / W)) / 2.0
        self.eval()

        K = self.n_extra_cond
        if K > 0:
            if modality is None:
                modality = np.zeros((T, K), np.float32)
            else:
                modality = np.asarray(modality, np.float32).reshape(-1, K)
                if modality.shape[0] != T:  # trim/zero-pad rows to T
                    fix = np.zeros((T, K), np.float32)
                    m = min(T, modality.shape[0])
                    fix[:m] = modality[:m]
                    modality = fix
        ctx_t = None
        if self.ctx_dim > 0 and ctx_vec is not None:
            ctx_t = torch.tensor(
                np.asarray(ctx_vec, np.float32).reshape(1, self.ctx_dim), device=dev
            )

        def _mod_win(s, e):
            """Modality slice [s:e] padded to W columns-stable (W,K)."""
            if K == 0:
                return None
            mw = modality[s:e]
            if len(mw) < W:
                mw = np.concatenate([mw, np.zeros((W - len(mw), K), np.float32)])
            return mw.astype(np.float32)

        def _win_stats(ts_w, code_w):
            obs_vals = ts_w[code_w == 1]
            if len(obs_vals) > 0:
                return np.array(
                    [
                        obs_vals.mean(),
                        obs_vals.std(),
                        (obs_vals < -0.47).mean(),
                        (obs_vals > 2.13).mean(),
                        len(obs_vals) / W,
                    ],
                    dtype=np.float32,
                )
            return np.zeros(5, dtype=np.float32)

        def _build_cond(base_w, ts_w, code_w):
            stats = _win_stats(ts_w, code_w)
            obs_w = (code_w == 1).astype(np.float32)
            gaps = compute_gap_distances(obs_w, W)
            linterp = compute_linterp_features(obs_w, ts_w, W)
            pchip_f = compute_pchip_features(obs_w, ts_w, W)
            slopes = compute_boundary_slope_features(obs_w, ts_w, W)
            accels = compute_boundary_accel_features(obs_w, ts_w, W)
            smooth_slopes = compute_smooth_slope_features(obs_w, ts_w, W)
            gap_ctx = compute_gap_context_features(obs_w, ts_w, W)
            return np.concatenate(
                [
                    base_w,
                    np.tile(stats, (W, 1)),
                    gaps,
                    linterp,
                    pchip_f,
                    slopes,
                    accels,
                    smooth_slopes,
                    gap_ctx,
                ],
                axis=1,
            )  # (W, 42)

        def _sliding_impute(cur_ts, cur_code_np, cur_cond):
            pred_acc = np.zeros(T, np.float64)
            weight_acc = np.zeros(T, np.float64)
            if T < W:
                pad = W - T
                ts_p = np.concatenate([cur_ts, np.zeros(pad, np.float32)])
                code_p = np.concatenate([cur_code_np, np.zeros(pad, np.int64)])
                base_p = np.concatenate(
                    [cur_cond, np.zeros((pad, N_COND_BASE), np.float32)]
                )
                cond_p = _build_cond(base_p, ts_p, code_p.astype(np.float32))
                mw = _mod_win(0, T)
                if mw is not None:
                    cond_p = np.concatenate([cond_p, mw], axis=1)
                x = torch.tensor(ts_p, device=dev).view(1, W, 1)
                oc = torch.tensor(code_p, device=dev, dtype=torch.long).view(1, W)
                pi = torch.arange(W, device=dev).view(1, W)
                ti = (torch.arange(W, device=dev) % 288).view(1, W)
                cd = torch.tensor(cond_p, device=dev, dtype=torch.float32).unsqueeze(0)
                pred_w = (
                    self._encode(x, oc, pi, ti, cd, None, ctx_t).squeeze().cpu().numpy()
                )
                pred_acc[:T] += pred_w[:T] * cos_w[:T]
                weight_acc[:T] += cos_w[:T]
            else:
                start = 0
                while True:
                    end = min(start + W, T)
                    ts_w = cur_ts[start:end]
                    code_w = cur_code_np[start:end]
                    base_w = cur_cond[start:end]
                    if len(ts_w) < W:
                        pad = W - len(ts_w)
                        ts_w = np.concatenate([ts_w, np.zeros(pad, np.float32)])
                        code_w = np.concatenate([code_w, np.zeros(pad, np.int64)])
                        base_w = np.concatenate(
                            [base_w, np.zeros((pad, N_COND_BASE), np.float32)]
                        )
                    cond_w = _build_cond(base_w, ts_w, code_w.astype(np.float32))
                    mw = _mod_win(start, end)
                    if mw is not None:
                        cond_w = np.concatenate([cond_w, mw], axis=1)
                    pos_idx_w = np.arange(W, dtype=np.int64)
                    tod_idx_w = (np.arange(start, start + W) % 288).astype(np.int64)
                    x = torch.tensor(ts_w, device=dev).view(1, W, 1)
                    oc = torch.tensor(code_w.astype(np.int64), device=dev).view(1, W)
                    pi = torch.tensor(pos_idx_w, device=dev).view(1, W)
                    ti = torch.tensor(tod_idx_w, device=dev).view(1, W)
                    cd = torch.tensor(
                        cond_w, device=dev, dtype=torch.float32
                    ).unsqueeze(0)
                    pred_w = (
                        self._encode(x, oc, pi, ti, cd, None, ctx_t)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
                    act = min(end, T) - start
                    pred_acc[start : start + act] += pred_w[:act] * cos_w[:act]
                    weight_acc[start : start + act] += cos_w[:act]
                    if end == T:
                        break
                    start += stride
            weight_acc = np.where(weight_acc > 0, weight_acc, 1.0)
            return pred_acc / weight_acc

        code_pass1 = mask.astype(np.int64)
        pred1 = _sliding_impute(ts, code_pass1, cond_full)
        result1 = np.where(mask.astype(bool), ts, pred1)
        if n_refinements == 0:
            return np.clip(result1, -6.0, 6.0).astype(np.float32)
        cur_result = result1.astype(np.float32)
        for _ in range(n_refinements):
            code_refine = np.where(mask.astype(bool), 1, 2).astype(np.int64)
            cond_refine = precompute_combined(cur_result, np.ones(T, dtype=bool))
            pred_refine = _sliding_impute(cur_result, code_refine, cond_refine)
            cur_result = np.where(mask.astype(bool), ts, pred_refine).astype(np.float32)
        return np.clip(cur_result, -6.0, 6.0).astype(np.float32)


class CGMV66Wrapper:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        self.model = CGMMAEV66(
            **{
                k: v
                for k, v in cfg.items()
                if k
                in ("window", "d_model", "n_heads", "n_layers", "ff_dim", "n_datasets")
            }
        )
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device
        self.model.to(torch.device(device))
        ep = ck.get("epoch", "?")
        val = ck.get("val_rmse", float("nan"))
        print(f"  CGM-MAE-v66 loaded (epoch {ep}, val={val:.2f} mg/dL)")

    def impute(self, full_ts, obs_mask):
        return self.model.impute(
            full_ts, obs_mask, device=str(self.device), stride=144, n_refinements=2
        )


STRATS = ["meal_post", "sleep", "ascending", "dipping", "combined"]


class CGMV66Dataset(Dataset):
    """8L, 3-pass MSE, N_COND=42 (37 v60 features + 5 gap-context features)."""

    def __init__(
        self,
        records,
        window=576,
        stride=144,
        seed=42,
        ts_mods=None,
        static_blocks=None,
        mask_mode="physio",
    ):
        from cgm_datasets.multimodal.modality_spec import (
            TS_MODALITIES,
            STATIC_BLOCKS,
            ts_width,
            ctx_width,
            build_mod_and_ctx,
        )

        self.window = window
        self.rng = np.random.default_rng(seed)
        # mask_mode="physio" (default) keeps the CGM physiological-strategy masking
        # unchanged. "realistic" uses generic gap augmentation (MCAR + random contiguous
        # gaps, varied length/ratio) for non-CGM signals where CGM masks are meaningless.
        self.mask_mode = mask_mode

        self.ts_mods = [m for m in TS_MODALITIES if ts_mods and m in ts_mods]
        self.static_blocks = [
            b for b in STATIC_BLOCKS if static_blocks and b in static_blocks
        ]
        self.n_extra_cond = ts_width(self.ts_mods)
        self.ctx_dim = ctx_width(self.static_blocks)
        self.ts_list = []
        self.mask_list = []
        self.windows = []
        self.dom_list = []
        self.mod_list = []
        self.ctx_list = []
        t0 = time.time()
        for ri, s in enumerate(records):
            ts = s["irg_ts"][:, 0].astype(np.float32)
            mask = s["irg_ts_mask"][:, 0].astype(bool)
            self.ts_list.append(ts)
            self.mask_list.append(mask)
            self.dom_list.append(int(s.get("domain_id", 0)))
            T = len(ts)
            # per-timestep modality block (T,K) + per-subject static ctx (S,), via the
            # shared spec helper so training and eval build identical layouts.
            mod, ctx = build_mod_and_ctx(s, self.ts_mods, self.static_blocks)
            self.mod_list.append(mod)
            self.ctx_list.append(ctx)
            for start in range(0, max(1, T - window), stride):
                self.windows.append((ri, start))
        print(
            f"  Done: {len(self.windows)} windows, K={self.n_extra_cond} "
            f"S={self.ctx_dim} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    def __len__(self):
        return len(self.windows)

    def window_domains(self):
        return np.array(
            [self.dom_list[ri] for (ri, _start) in self.windows], dtype=np.int64
        )

    def __getitem__(self, idx):
        ri, start = self.windows[idx]
        ts_full = self.ts_list[ri]
        mask_full = self.mask_list[ri]
        W = self.window
        end = start + W
        ts = ts_full[start:end].copy()
        orig = mask_full[start:end].copy()
        n_actual = len(ts)
        if n_actual < W:
            pad = W - n_actual
            ts = np.concatenate([ts, np.zeros(pad, np.float32)])
            orig = np.concatenate([orig, np.zeros(pad, bool)])
        if self.mask_mode == "realistic":
            from utils.missingness_mechanisms import create_realistic_mask

            tr = float(self.rng.uniform(0.05, 0.35))  # cover the eval missingness range
            em, tm = create_realistic_mask(
                ts.reshape(-1, 1).copy(),
                orig.reshape(-1, 1).copy(),
                target_ratio=tr,
                seed=int(self.rng.integers(0, 2**31)),
            )
        else:
            strat = STRATS[self.rng.integers(len(STRATS))]
            em, tm = create_physiological_mask(
                ts.reshape(-1, 1).copy(),
                orig.reshape(-1, 1).copy(),
                strategy=strat,
                target_ratio=0.20,
                seed=int(self.rng.integers(0, 2**31)),
            )
        eval_mask = em[:, 0].astype(np.float32)
        target_mask = tm[:, 0].astype(bool)
        mask_with_gap = mask_full.copy()
        gap_local = orig & ~eval_mask.astype(bool)
        gap_global = start + np.where(gap_local[:n_actual])[0]
        gap_global = gap_global[gap_global < len(mask_full)]
        if len(gap_global):
            mask_with_gap[gap_global] = False
        cond_base = precompute_combined(ts_full, mask_with_gap)[start:end]
        if len(cond_base) < W:
            cond_base = np.concatenate(
                [cond_base, np.zeros((W - len(cond_base), N_COND_BASE), np.float32)]
            )
        obs_in_win = ts[eval_mask.astype(bool)]
        if len(obs_in_win) > 0:
            win_stats = np.array(
                [
                    obs_in_win.mean(),
                    obs_in_win.std(),
                    (obs_in_win < -0.47).mean(),
                    (obs_in_win > 2.13).mean(),
                    len(obs_in_win) / W,
                ],
                dtype=np.float32,
            )
        else:
            win_stats = np.zeros(5, dtype=np.float32)
        win_block = np.tile(win_stats, (W, 1))
        gap_dists = compute_gap_distances(eval_mask, W)
        linterp = compute_linterp_features(eval_mask, ts, W)
        pchip_feat = compute_pchip_features(eval_mask, ts, W)
        slopes = compute_boundary_slope_features(eval_mask, ts, W)
        accels = compute_boundary_accel_features(eval_mask, ts, W)
        smooth_slopes = compute_smooth_slope_features(eval_mask, ts, W)
        gap_ctx = compute_gap_context_features(eval_mask, ts, W)  # (W, 5) NEW
        cond = np.concatenate(
            [
                cond_base,
                win_block,
                gap_dists,
                linterp,
                pchip_feat,
                slopes,
                accels,
                smooth_slopes,
                gap_ctx,
            ],
            axis=1,
        )  # (W, 42)

        # NOT masked by the CGM gap) -> (W, 42+K); build static ctx vector (S,).
        if self.n_extra_cond:
            mod = self.mod_list[ri][start : start + W]
            if len(mod) < W:
                mod = np.concatenate(
                    [mod, np.zeros((W - len(mod), self.n_extra_cond), np.float32)]
                )
            cond = np.concatenate([cond, mod], axis=1)  # (W, 42+K)
        ctx = self.ctx_list[ri] if self.ctx_dim else np.zeros(0, np.float32)
        pos_idx = np.arange(W, dtype=np.int64)
        tod_idx = (np.arange(start, start + W) % 288).astype(np.int64)
        dom = self.dom_list[ri]
        return (
            torch.tensor(ts * eval_mask, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(eval_mask, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(tod_idx, dtype=torch.long),
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(ts, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(target_mask, dtype=torch.bool).unsqueeze(-1),
            torch.tensor(dom, dtype=torch.long),
            torch.tensor(ctx, dtype=torch.float32),
        )


def make_train_loader(ds_train, *, batch_size, mix="equal", num_workers=4):
    """DataLoader with a domain-balanced sampler over the dataset's windows."""
    from cgm_datasets.pooled import make_balanced_sampler

    sampler = make_balanced_sampler(ds_train.window_domains(), mix=mix)
    return DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def warm_start_from(model, init_from):
    """Load weights from a checkpoint into `model`, skipping shape-mismatched keys.

    `strict=False` alone only ignores missing/unexpected keys, NOT shape mismatches.
    When finetuning aireadi-only (n_datasets=1) from a pooled checkpoint (n_datasets=N),
    `dom_emb` shapes differ; those keys are dropped so they keep their zero-init ,
    AI-READI is domain 0, so a zero `dom_emb` reproduces the published behavior.

    Returns the checkpoint's `val_rmse` (float) or None if absent/NaN.
    """
    ck = torch.load(init_from, map_location="cpu", weights_only=False)
    state = ck["model"]
    cur = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in cur and v.shape == cur[k].shape}
    # MULTIMODAL cond_proj column-copy surgery: when the model adds K extra cond
    # columns, cond_proj.0.weight grows (64,42)->(64,42+K). A plain shape filter would
    # SKIP it (leaving random init) and break parity. Instead copy the trained 42
    # columns and zero the new K, so the modality contributes nothing at init.
    ckw = "cond_proj.0.weight"
    if ckw in state and ckw in cur and state[ckw].shape != cur[ckw].shape:
        old = state[ckw]
        new = cur[ckw].clone()
        c = min(old.shape[1], new.shape[1])
        if old.shape[0] == new.shape[0] and new.shape[1] >= old.shape[1]:
            new.zero_()
            new[:, :c] = old[:, :c]
            filtered[ckw] = new
            print(
                f"  Warm-start: cond_proj surgery {tuple(old.shape)}->{tuple(new.shape)} "
                f"(copied {c} cols, zeroed {new.shape[1]-c})"
            )
    skipped = [k for k in state if k not in filtered]
    info = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"  Warm-start: skipped shape-mismatched keys (kept init): {skipped}")
    if info.missing_keys:
        print(f"  Warm-start: missing keys left at init: {info.missing_keys}")
    val = ck.get("val_rmse", float("nan"))
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    from cgm_datasets.pooled import tag_domains

    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    by_name_train, by_name_val = {}, {}
    for name in dataset_names:
        tr, va, _, _ = load_dataset_splits(name, prepared_dir=args.data_dir)
        by_name_train[name] = tr
        by_name_val[name] = va
    train_r, dom_names = tag_domains(by_name_train)
    val_r, _ = tag_domains(by_name_val)
    n_datasets = len(dom_names)
    print(
        f"  stage={args.stage}  datasets={dom_names}  train={len(train_r)} val={len(val_r)}"
    )
    print("Building train dataset (W=576, 42-feat, 3-pass MSE, 8L, v60+gap-ctx)...")
    ds_train = CGMV66Dataset(train_r, window=args.window, stride=144, seed=42)
    print("Building val dataset ...")
    ds_val = CGMV66Dataset(val_r, window=args.window, stride=args.window, seed=99)
    print(f"  train windows={len(ds_train)}  val windows={len(ds_val)}")
    if args.stage == "finetune":
        dl_train = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    else:
        dl_train = make_train_loader(
            ds_train, batch_size=args.batch_size, mix=args.mix, num_workers=4
        )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model = CGMMAEV66(
        window=args.window,
        d_model=128,
        n_heads=8,
        n_layers=8,
        ff_dim=512,
        n_datasets=n_datasets,
    ).to(device)
    save_path = p("cgm_mae_v66_imputer.pt")
    snap_dir = p("checkpoints_v66")
    os.makedirs(snap_dir, exist_ok=True)
    SNAP_EPOCHS = {10, 20, 30, 40, 50}
    best_val = args.min_val

    if args.init_from and os.path.exists(args.init_from):
        incoming_val = warm_start_from(model, args.init_from)
        if incoming_val is not None:
            best_val = min(best_val, incoming_val)  # don't overwrite a better ckpt
        print(f"  Warm-started from {args.init_from} (val={incoming_val})")
    else:
        print(f"  Training from scratch (init_from={args.init_from!r} not found)")

    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print(
        f"  LR schedule: ReduceLROnPlateau(start={args.lr:.0e}, patience=5, factor=0.5)"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, verbose=True
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss1 = 0.0
        tr_loss2 = 0.0
        tr_loss3 = 0.0
        tr_cnt = 0
        for obs, msk, pos, tod, cond, gt, tgt, dom, ctx in dl_train:
            obs, msk = obs.to(device), msk.to(device)
            pos, tod = pos.to(device), tod.to(device)
            cond, gt = cond.to(device), gt.to(device)
            tgt = tgt.to(device)
            dom = dom.to(device)
            ctx = ctx.to(device)
            pred1, pred2, pred3 = model(
                obs, msk, pos, tod, cond, three_pass=True, dom_idx=dom, ctx_vec=ctx
            )
            tgt_f = tgt.float()
            n_tgt = tgt_f.sum() + 1e-8
            l1 = ((pred1 - gt) ** 2 * tgt_f).sum() / n_tgt
            l2 = ((pred2 - gt) ** 2 * tgt_f).sum() / n_tgt
            l3 = ((pred3 - gt) ** 2 * tgt_f).sum() / n_tgt
            loss = PASS1_WEIGHT * l1 + PASS2_WEIGHT * l2 + PASS3_WEIGHT * l3
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int(tgt.sum())
            tr_loss1 += float(l1) * n
            tr_loss2 += float(l2) * n
            tr_loss3 += float(l3) * n
            tr_cnt += n

        model.eval()
        vl_loss1 = 0.0
        vl_loss2 = 0.0
        vl_loss3 = 0.0
        vl_cnt = 0
        with torch.no_grad():
            for obs, msk, pos, tod, cond, gt, tgt, dom, ctx in dl_val:
                obs, msk = obs.to(device), msk.to(device)
                pos, tod = pos.to(device), tod.to(device)
                cond, gt = cond.to(device), gt.to(device)
                tgt = tgt.to(device)
                dom = dom.to(device)
                ctx = ctx.to(device)
                p1, p2, p3 = model(
                    obs, msk, pos, tod, cond, three_pass=True, dom_idx=dom, ctx_vec=ctx
                )
                tgt_f = tgt.float()
                vl_loss1 += float(((p1 - gt) ** 2 * tgt_f).sum())
                vl_loss2 += float(((p2 - gt) ** 2 * tgt_f).sum())
                vl_loss3 += float(((p3 - gt) ** 2 * tgt_f).sum())
                vl_cnt += int(tgt.sum())

        tr_r1 = np.sqrt(tr_loss1 / max(tr_cnt, 1)) * CGM_STD
        tr_r2 = np.sqrt(tr_loss2 / max(tr_cnt, 1)) * CGM_STD
        tr_r3 = np.sqrt(tr_loss3 / max(tr_cnt, 1)) * CGM_STD
        vl_r1 = np.sqrt(vl_loss1 / max(vl_cnt, 1)) * CGM_STD
        vl_r2 = np.sqrt(vl_loss2 / max(vl_cnt, 1)) * CGM_STD
        vl_r3 = np.sqrt(vl_loss3 / max(vl_cnt, 1)) * CGM_STD
        lr_now = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={tr_r1:.2f}/{tr_r2:.2f}/{tr_r3:.2f}  "
            f"val={vl_r1:.2f}/{vl_r2:.2f}/{vl_r3:.2f} mg/dL  "
            f"lr={lr_now:.1e}  ({elapsed:.0f}s)",
            flush=True,
        )
        sched.step(vl_r3)
        ckpt_data = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_rmse": vl_r3,
            "val_rmse_pass1": vl_r1,
            "val_rmse_pass2": vl_r2,
            "config": {
                "window": args.window,
                "d_model": 128,
                "n_heads": 8,
                "n_layers": 8,
                "ff_dim": 512,
                "n_datasets": n_datasets,
            },
        }
        if vl_r3 < best_val:
            best_val = vl_r3
            torch.save(ckpt_data, save_path)
            print(f"  → Saved best (pass3 val={best_val:.2f})", flush=True)
        if epoch in SNAP_EPOCHS:
            snap_path = os.path.join(snap_dir, f"cgm_mae_v66_ep{epoch:02d}.pt")
            torch.save(ckpt_data, snap_path)
            print(f"  → Snapshot: {snap_path}", flush=True)

    print(f"\nDone. Best pass-3 val RMSE = {best_val:.2f} mg/dL → {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window", type=int, default=576)
    ap.add_argument("--min_val", type=float, default=999.0)
    ap.add_argument(
        "--datasets", default="aireadi", help="comma list, e.g. aireadi,hupa_ucm,ohio"
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="back-compat single dataset; if set, overrides --datasets",
    )
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--mix", default="equal", choices=["equal", "sqrt", "proportional"])
    ap.add_argument("--stage", default="pretrain", choices=["pretrain", "finetune"])
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.dataset:
        args.datasets = args.dataset
    train(args)
