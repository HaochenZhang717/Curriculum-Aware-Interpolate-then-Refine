"""MIMIC single-vital downstream recovery with full imputation suite and two readouts."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (ROOT, os.path.join(ROOT, "baselines", "statistical")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baselines.toye.toye_baselines import (  # noqa: E402
    impute_fourier,
    impute_knn,
    impute_locf,
    impute_mice,
    impute_missforest,
    impute_mode,
)
from baselines_autoregressive import pchip_impute  # noqa: E402
from cgm_datasets import load_dataset_splits  # noqa: E402
from experiments.evaluation.mimic_singlevital_downstream import (  # noqa: E402
    complete,
    derive_labels,
    prognostic_features,
    to_native,
)
from simple import forward_fill, linear_interp  # noqa: E402
from sota_baselines import (  # noqa: E402
    akima_interp,
    ewma_impute,
    savitzky_golay,
    spline_cubic,
)
from utils.missingness_mechanisms import (  # noqa: E402
    create_mcar_mask,
    create_nmar_mask,
    native_gap_durations,
)

LABELS = ("mortality", "sepsis", "hf")
SUPPORTED_MECHS = ("mcar", "nmar")
ALL_METHODS = (
    "real",
    "mean",
    "mode",
    "forward_fill",
    "linear",
    "pchip",
    "akima",
    "spline",
    "savgol",
    "ewma",
    "knn",
    "mice",
    "missforest",
    "locf",
    "fourier",
    "cair",
    "cair_mm",
)
DEFAULT_METHODS = (
    "real,mean,mode,forward_fill,linear,pchip,akima,spline,savgol,"
    "knn,mice,missforest,cair,cair_mm"
)
DEFAULT_MECHS = "mcar,nmar"
DEFAULT_RATES = "0.1,0.2,0.3"
DEFAULT_TS_MODS = ["hr", "resp", "spo2"]
DEFAULT_GAP_DURS = [3, 6, 9, 12]
GBM_SEEDS = 5
CNN_SEEDS = 3
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 256
PRED_BATCH_SIZE = 512


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_methods(text: str) -> list[str]:
    methods = []
    for method in _parse_csv(text):
        if method not in methods:
            methods.append(method)
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(
            f"unsupported method(s): {unknown}; choices={list(ALL_METHODS)}"
        )
    if not methods:
        raise SystemExit("no methods requested")
    return methods


def _parse_mechs(text: str) -> list[str]:
    mechs = []
    for mech in _parse_csv(text):
        if mech not in mechs:
            mechs.append(mech)
    unknown = [m for m in mechs if m not in SUPPORTED_MECHS]
    if unknown:
        raise SystemExit(
            f"unsupported mech(s): {unknown}; choices={list(SUPPORTED_MECHS)}"
        )
    if not mechs:
        raise SystemExit("no missingness mechanisms requested")
    return mechs


def _parse_rates(text: str) -> list[float]:
    rates = [float(x) for x in _parse_csv(text)]
    if not rates:
        raise SystemExit("no rates requested")
    return rates


def _thresholds_for(dataset: str) -> tuple[float, float]:
    if "hr" in dataset.lower():
        return 50.0, 100.0
    return 65.0, 100.0


def _resolve_device(requested: str) -> str:
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[note] CUDA unavailable; using cpu instead of {requested}", flush=True)
        return "cpu"
    return requested


def _load_torch_checkpoint(path: str):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def zspace_impute(method: str, z: np.ndarray, obsb: np.ndarray) -> np.ndarray:
    """Return a fully-imputed (T,) z-space series for the requested baseline."""
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    obsb = np.asarray(obsb).reshape(-1).astype(bool)
    if z.shape[0] != obsb.shape[0]:
        raise ValueError(f"shape mismatch: z={z.shape} obsb={obsb.shape}")
    if not obsb.any():
        return np.zeros_like(z, dtype=np.float32)

    seen = np.where(obsb, z, np.nan).astype(np.float32)
    obsf = obsb.astype(np.float32)
    if method == "forward_fill":
        pred = forward_fill(seen, obsf)
    elif method == "linear":
        pred = linear_interp(seen, obsf)
    elif method == "pchip":
        pred = pchip_impute(seen, obsb)
    elif method == "akima":
        pred = akima_interp(seen, obsf)
    elif method == "spline":
        pred = spline_cubic(seen, obsf)
    elif method == "savgol":
        pred = savitzky_golay(seen, obsf)
    elif method == "ewma":
        pred = ewma_impute(seen.reshape(-1, 1), obsf.reshape(-1, 1))
    elif method == "knn":
        pred = impute_knn(seen, obsf)
    elif method == "mice":
        pred = impute_mice(seen, obsf)
    elif method == "missforest":
        pred = impute_missforest(seen, obsf)
    elif method == "mode":
        pred = impute_mode(seen, obsf)
    elif method == "locf":
        pred = impute_locf(seen, obsf)
    elif method == "fourier":
        pred = impute_fourier(seen, obsf)
    else:
        raise ValueError(f"zspace_impute does not handle method '{method}'")

    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    if pred.shape[0] != z.shape[0]:
        raise ValueError(f"bad imputer output shape for '{method}': {pred.shape}")
    if not np.all(np.isfinite(pred)):
        fallback = np.asarray(linear_interp(seen, obsf), dtype=np.float32).reshape(-1)
        pred = np.where(np.isfinite(pred), pred, fallback)
    return pred.astype(np.float32)


def _label_guard(y: np.ndarray) -> bool:
    y = np.asarray(y, dtype=np.int64)
    return len(np.unique(y)) >= 2 and int(y.sum()) >= 20


def fit_gbm_reference(
    train_recs, labels, mean: float, std: float, lo: float, hi: float
):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    feats = []
    y_by_label = {label: [] for label in LABELS}
    for rec in train_recs:
        sid = rec.get("stay_id")
        if sid is None or int(sid) not in labels:
            continue
        native, obs = to_native(rec, mean, std)
        feats.append(prognostic_features(complete(native, obs), lo, hi))
        rec_labels = labels[int(sid)]
        for label in LABELS:
            y_by_label[label].append(int(rec_labels[label]))

    if not feats:
        return {}, None

    X = np.vstack(feats).astype(np.float32)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = {}
    for label in LABELS:
        y = np.asarray(y_by_label[label], dtype=np.int64)
        if not _label_guard(y):
            print(
                f"[skip] label={label} positives={int(y.sum())} classes={np.unique(y).tolist()}",
                flush=True,
            )
            continue
        models[label] = [
            HistGradientBoostingClassifier(
                max_iter=300,
                max_depth=4,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=seed,
            ).fit(Xs, y)
            for seed in range(GBM_SEEDS)
        ]
        print(
            f"[train] gbm label={label} n={len(y)} pos={int(y.sum())} seeds={GBM_SEEDS}",
            flush=True,
        )
    return models, scaler


def _predict_gbm_probs(estimators, Xs: np.ndarray) -> np.ndarray:
    probs = []
    for estimator in estimators:
        p = estimator.predict_proba(Xs)
        pos_idx = int(np.where(estimator.classes_ == 1)[0][0])
        probs.append(p[:, pos_idx])
    return np.mean(probs, axis=0).astype(np.float32)


class Small1DCNN(nn.Module):  # pragma: no cover - exercised in acceptance smoke
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
        )
        self.out_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.out_pool(x).squeeze(-1)
        return self.head(x)


def _train_cnn_member(x_cpu, y_np: np.ndarray, device: str, seed: int):
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = Small1DCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(y_np.sum())
    neg = float(len(y_np) - pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [neg / max(pos, 1.0)], device=device, dtype=torch.float32
        )
    )
    y_cpu = torch.from_numpy(y_np.astype(np.float32).reshape(-1, 1))

    model.train()
    for _ in range(CNN_EPOCHS):
        order = rng.permutation(len(y_np))
        for start in range(0, len(order), CNN_BATCH_SIZE):
            batch_idx = order[start : start + CNN_BATCH_SIZE]
            idx = torch.from_numpy(batch_idx.astype(np.int64))
            xb = x_cpu.index_select(0, idx).to(device)
            yb = y_cpu.index_select(0, idx).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def fit_cnn_reference(
    train_recs, labels, mean: float, std: float, lo: float, hi: float, device: str
):
    del lo, hi  # thresholds are only used by the hand-feature readout.

    traces = []
    y_by_label = {label: [] for label in LABELS}
    for rec in train_recs:
        sid = rec.get("stay_id")
        if sid is None or int(sid) not in labels:
            continue
        native, obs = to_native(rec, mean, std)
        full_native = complete(native, obs)
        traces.append(((full_native - mean) / std).astype(np.float32))
        rec_labels = labels[int(sid)]
        for label in LABELS:
            y_by_label[label].append(int(rec_labels[label]))

    if not traces:
        return {}

    x_cpu = np.stack(traces).astype(np.float32)
    x_cpu = x_cpu[:, None, :]
    import torch

    x_cpu = torch.from_numpy(x_cpu)
    models = {}
    for label in LABELS:
        y = np.asarray(y_by_label[label], dtype=np.float32)
        if not _label_guard(y):
            print(
                f"[skip] label={label} positives={int(y.sum())} classes={np.unique(y).tolist()}",
                flush=True,
            )
            continue
        models[label] = [
            _train_cnn_member(x_cpu, y, device, seed) for seed in range(CNN_SEEDS)
        ]
        print(
            f"[train] cnn label={label} n={len(y)} pos={int(y.sum())} seeds={CNN_SEEDS}",
            flush=True,
        )
    return models


def _predict_cnn_probs(ensemble, z_traces: np.ndarray, device: str) -> np.ndarray:
    z_traces = np.asarray(z_traces, dtype=np.float32)
    if z_traces.size == 0:
        return np.zeros(0, dtype=np.float32)
    if z_traces.ndim != 2:
        raise ValueError(f"expected (N,T) z_traces, got {z_traces.shape}")
    x_cpu = torch.from_numpy(z_traces[:, None, :])
    logits_by_seed = []
    with torch.no_grad():
        for model in ensemble:
            seed_logits = []
            for start in range(0, len(z_traces), PRED_BATCH_SIZE):
                xb = x_cpu[start : start + PRED_BATCH_SIZE].to(device)
                seed_logits.append(model(xb).squeeze(1).cpu().numpy())
            logits_by_seed.append(np.concatenate(seed_logits, axis=0))
    mean_logits = np.mean(logits_by_seed, axis=0)
    mean_logits = np.clip(mean_logits, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-mean_logits))).astype(np.float32)


def _load_cair_imputers(
    methods: list[str], dataset: str, ckpt_suffix: str, device: str
):
    methods = list(methods)
    cair = None
    cair_mm = None
    ts_mods = list(DEFAULT_TS_MODS)
    ckpt_dir = os.path.join(ROOT, "method_checkpoints", f"cair_{dataset}{ckpt_suffix}")

    if "cair" in methods:
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "uni_seed*.pt")))
        if paths:
            from methods.cair import CAIRImputer

            cair = CAIRImputer(ckpt_paths=paths, device=device)
        else:
            print(f"[skip] method=cair no uni_seed*.pt under {ckpt_dir}", flush=True)
            methods = [m for m in methods if m != "cair"]

    if "cair_mm" in methods:
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "mm_seed*.pt")))
        if paths:
            from methods.cair import CAIRImputer

            cair_mm = CAIRImputer(ckpt_paths=paths, device=device)
            cfg = _load_torch_checkpoint(paths[0]).get("config", {})
            ts_mods = cfg.get("ts_mods", ts_mods) or ts_mods
        else:
            print(f"[skip] method=cair_mm no mm_seed*.pt under {ckpt_dir}", flush=True)
            methods = [m for m in methods if m != "cair_mm"]
    return methods, cair, cair_mm, ts_mods


def _compute_auroc(y: np.ndarray, probs: np.ndarray):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)
    if len(y) == 0 or len(probs) != len(y) or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, probs))


def _score_condition(
    clf: str,
    models,
    scaler,
    filled_by_method: dict[str, list[np.ndarray]],
    y_by_label: dict[str, list[int]],
    methods: list[str],
    lo: float,
    hi: float,
    mean: float,
    std: float,
    device: str,
    mech: str,
    rate: float,
):
    results = []
    if clf == "gbm":
        Xs_by_method = {}
        for method in methods:
            traces = filled_by_method.get(method) or []
            if not traces:
                continue
            X = np.vstack(
                [prognostic_features(trace, lo, hi) for trace in traces]
            ).astype(np.float32)
            Xs_by_method[method] = scaler.transform(X)
        for label, ensemble in models.items():
            y = np.asarray(y_by_label[label], dtype=np.int64)
            for method in methods:
                Xs = Xs_by_method.get(method)
                probs = (
                    _predict_gbm_probs(ensemble, Xs) if Xs is not None else np.array([])
                )
                results.append(
                    {
                        "label": label,
                        "method": method,
                        "mech": mech,
                        "rate": float(rate),
                        "auroc": _compute_auroc(y, probs),
                        "n": int(len(y)),
                    }
                )
        return results

    z_by_method = {}
    for method in methods:
        traces = filled_by_method.get(method) or []
        if not traces:
            continue
        native = np.stack(traces).astype(np.float32)
        z_by_method[method] = ((native - mean) / std).astype(np.float32)
    for label, ensemble in models.items():
        y = np.asarray(y_by_label[label], dtype=np.int64)
        for method in methods:
            z_traces = z_by_method.get(method)
            probs = (
                _predict_cnn_probs(ensemble, z_traces, device)
                if z_traces is not None
                else np.array([])
            )
            results.append(
                {
                    "label": label,
                    "method": method,
                    "mech": mech,
                    "rate": float(rate),
                    "auroc": _compute_auroc(y, probs),
                    "n": int(len(y)),
                }
            )
    return results


def evaluate(
    eval_recs,
    labels,
    clf: str,
    models,
    scaler,
    mean: float,
    std: float,
    lo: float,
    hi: float,
    methods: list[str],
    mechs: list[str],
    rates: list[float],
    device: str,
    cair=None,
    cair_mm=None,
    ts_mods=None,
):
    active_labels = [label for label in LABELS if label in models]
    if not active_labels:
        return []

    need_mm = cair_mm is not None and "cair_mm" in methods
    if need_mm:
        from cgm_datasets.multimodal.modality_spec import build_mod_and_ctx

    results = []
    for mech_idx, mech in enumerate(mechs):
        for rate_idx, rate in enumerate(rates):
            print(f"[eval] mech={mech} rate={rate:.3f}", flush=True)
            filled_by_method = {method: [] for method in methods}
            y_by_label = {label: [] for label in active_labels}
            for rec_idx, rec in enumerate(eval_recs):
                sid = rec.get("stay_id")
                if sid is None or int(sid) not in labels:
                    continue
                native, obs = to_native(rec, mean, std)
                reference = complete(native, obs)
                z = np.asarray(rec["irg_ts"][:, 0], dtype=np.float32)
                orig_mask = np.asarray(rec["irg_ts_mask"], dtype=np.float32)
                gap_durs = native_gap_durations(orig_mask) or DEFAULT_GAP_DURS
                seed = 9000 + rec_idx * 100 + mech_idx * 10 + rate_idx
                if mech == "mcar":
                    _eval_mask, target = create_mcar_mask(
                        rec["irg_ts"], rec["irg_ts_mask"], percent=rate, seed=seed
                    )
                else:
                    _eval_mask, target = create_nmar_mask(
                        rec["irg_ts"],
                        rec["irg_ts_mask"],
                        percent=rate,
                        gap_durations=gap_durs,
                        seed=seed,
                        mean_mgdl=mean,
                        std_mgdl=std,
                        lo_mgdl=lo,
                        hi_mgdl=hi,
                    )
                tgt = np.asarray(target[:, 0]).astype(bool)
                if not tgt.any():
                    continue

                obsb = np.asarray(orig_mask[:, 0]).astype(bool) & (~tgt)
                mod = None
                ctx = None
                if need_mm:
                    mod, ctx = build_mod_and_ctx(rec, ts_mods, [])

                for method in methods:
                    filled = reference.copy()
                    if method == "real":
                        pass
                    elif method == "mean":
                        fill = (
                            float(reference[obsb].mean())
                            if obsb.any()
                            else float(reference.mean())
                        )
                        filled[tgt] = fill
                    elif method in ("cair", "cair_mm"):
                        imputer = cair if method == "cair" else cair_mm
                        if imputer is None:
                            continue
                        z_seen = z.copy()
                        z_seen[~obsb] = 0.0
                        if method == "cair_mm":
                            pred_z = imputer.impute(
                                z_seen,
                                obsb.astype(np.float32),
                                modality=mod,
                                ctx_vec=ctx,
                            )
                        else:
                            pred_z = imputer.impute(z_seen, obsb.astype(np.float32))
                        pred_native = (
                            np.asarray(pred_z, dtype=np.float32).reshape(-1) * std
                            + mean
                        )
                        filled[tgt] = pred_native[tgt]
                    else:
                        pred_z = zspace_impute(method, z, obsb)
                        pred_native = pred_z * std + mean
                        filled[tgt] = pred_native[tgt]
                    filled_by_method[method].append(filled.astype(np.float32))

                rec_labels = labels[int(sid)]
                for label in active_labels:
                    y_by_label[label].append(int(rec_labels[label]))
                if rec_idx % 50 == 0:
                    print(f"  record {rec_idx}/{len(eval_recs)}", flush=True)

            results.extend(
                _score_condition(
                    clf=clf,
                    models=models,
                    scaler=scaler,
                    filled_by_method=filled_by_method,
                    y_by_label=y_by_label,
                    methods=methods,
                    lo=lo,
                    hi=hi,
                    mean=mean,
                    std=std,
                    device=device,
                    mech=mech,
                    rate=rate,
                )
            )
    return results


def print_tables(
    results: list[dict], methods: list[str], mechs: list[str], rates: list[float]
):
    lookup = {
        (row["label"], row["method"], row["mech"], float(row["rate"])): row
        for row in results
    }
    labels = [
        label for label in LABELS if any(row["label"] == label for row in results)
    ]
    for label in labels:
        for mech in mechs:
            print(f"\n[{label}] mech={mech}", flush=True)
            header = "method".ljust(18) + "".join(f"{rate:>10.2f}" for rate in rates)
            print(header, flush=True)
            for method in methods:
                row = method.ljust(18)
                for rate in rates:
                    hit = lookup.get((label, method, mech, float(rate)))
                    if hit and hit["auroc"] is not None:
                        row += f"{hit['auroc']:>10.3f}"
                    else:
                        row += f"{'-':>10}"
                print(row, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mimic_abp")
    ap.add_argument(
        "--source_pkl", default=str(DATA_ROOT / "raw" / "mimic3_waveform_abp.pkl")
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--n_train", type=int, default=0, help="0=all")
    ap.add_argument("--eval_split", default="test", choices=("test", "valtest"))
    ap.add_argument("--clf", default="gbm", choices=("gbm", "cnn"))
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--mechs", default=DEFAULT_MECHS)
    ap.add_argument("--rates", default=DEFAULT_RATES)
    ap.add_argument("--ckpt_suffix", default="_realistic")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    methods = _parse_methods(args.methods)
    mechs = _parse_mechs(args.mechs)
    rates = _parse_rates(args.rates)
    device = _resolve_device(args.device)

    train, val, test, meta = load_dataset_splits(args.dataset)
    mean = float(meta.glucose_mean_mgdl)
    std = float(meta.glucose_std_mgdl)
    lo, hi = _thresholds_for(args.dataset)
    if args.n_train:
        train = train[: args.n_train]
    eval_recs = list(test) if args.eval_split == "test" else list(val) + list(test)
    if args.n_test:
        eval_recs = eval_recs[: args.n_test]
    labels = derive_labels(args.source_pkl)
    print(
        f"[data] dataset={args.dataset} clf={args.clf} train={len(train)} eval={len(eval_recs)} "
        f"thr=({lo},{hi}) device={device}",
        flush=True,
    )

    if args.clf == "gbm":
        models, scaler = fit_gbm_reference(train, labels, mean, std, lo, hi)
    else:
        models = fit_cnn_reference(train, labels, mean, std, lo, hi, device)
        scaler = None
    active_labels = [label for label in LABELS if label in models]
    print(f"[train] active_labels={active_labels}", flush=True)
    if not active_labels:
        out = {"dataset": args.dataset, "clf": args.clf, "results": []}
        if args.out:
            out_dir = os.path.dirname(os.path.abspath(args.out))
            os.makedirs(out_dir, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
        return

    methods, cair, cair_mm, ts_mods = _load_cair_imputers(
        methods, args.dataset, args.ckpt_suffix, device
    )
    print(f"[eval] methods={methods} mechs={mechs} rates={rates}", flush=True)
    results = evaluate(
        eval_recs=eval_recs,
        labels=labels,
        clf=args.clf,
        models=models,
        scaler=scaler,
        mean=mean,
        std=std,
        lo=lo,
        hi=hi,
        methods=methods,
        mechs=mechs,
        rates=rates,
        device=device,
        cair=cair,
        cair_mm=cair_mm,
        ts_mods=ts_mods,
    )
    print_tables(results, methods, mechs, rates)

    out = {"dataset": args.dataset, "clf": args.clf, "results": results}
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[out] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
