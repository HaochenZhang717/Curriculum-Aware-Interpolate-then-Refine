#!/usr/bin/env python3
"""Build the per-subject static-context blocks and write them into the mm split."""

from __future__ import annotations

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT

import glob
import json
import os
import pickle
import time

import numpy as np

D = str(DATA_ROOT)
MM = D + "/aireadi_cgm_mm"
ECG_MANIFEST = D + "/cardiac_ecg/manifest.tsv"
PARTICIPANTS = D + "/participants.tsv"
ECG_COLS = ["Rate", "PR", "QRSD", "QT", "QTc", "P", "QRS", "T"]
RET = {  # type -> glob pattern (per-pid filled in)
    "cfp": "retinal_photography/cfp/*/{pid}/*.dcm",
    "oct": "retinal_oct/structural_oct/*/{pid}/*.dcm",
    "octa": "retinal_octa/enface/*/{pid}/*.dcm",
    "flio": "retinal_flio/flio/*/{pid}/*.dcm",
}


def load_clinical():
    import pandas as pd

    df = pd.read_csv(PARTICIPANTS, sep="\t", dtype=str)
    out = {}
    for _, r in df.iterrows():
        try:
            age = float(r.get("age", "nan"))
        except ValueError:
            age = np.nan
        out[str(r["person_id"])] = age
    return out  # pid -> age


def load_ecg_scalars():
    import pandas as pd

    df = pd.read_csv(ECG_MANIFEST, sep="\t")
    out = {}
    for _, r in df.iterrows():
        vals = []
        for c in ECG_COLS:
            try:
                vals.append(float(r[c]))
            except (ValueError, TypeError, KeyError):
                vals.append(np.nan)
        out[str(r["person_id"])] = np.array(vals, np.float32)
    return out  # pid -> (8,)


def _to_chw518(arr):
    """Any DICOM pixel_array -> (3,518,518) float in [0,1]."""
    import torch
    import torch.nn.functional as F

    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 3:  # color HWC
        t = np.transpose(a, (2, 0, 1))
    elif a.ndim == 2:  # grayscale
        t = np.repeat(a[None], 3, axis=0)
    elif a.ndim == 3:  # volume (S,H,W) -> middle slice
        s = a[a.shape[0] // 2]
        t = np.repeat(s[None], 3, axis=0)
    elif a.ndim == 4:  # (S,H,W,C?) -> middle
        s = a[a.shape[0] // 2]
        t = (
            np.transpose(s, (2, 0, 1))
            if s.shape[-1] == 3
            else np.repeat(s[None], 3, axis=0)
        )
    else:
        raise ValueError(f"bad shape {a.shape}")
    t = torch.tensor(t).float()
    mx = float(t.max()) or 1.0
    t = (t / mx).unsqueeze(0)
    return F.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False)[0]


def encode_retinal(pid, model, device):
    """Return (1536,) concat of 4 DINOv2 type-embeddings + present flag (0/1)."""
    import pydicom
    import torch

    embs = []
    any_present = 0.0
    for typ in ("cfp", "oct", "octa", "flio"):
        fs = glob.glob(os.path.join(D, RET[typ].format(pid=pid)))
        e = np.zeros(384, np.float32)
        for f in fs[:1]:  # 1 image per type to bound IO
            try:
                arr = pydicom.dcmread(f).pixel_array
                x = _to_chw518(arr).unsqueeze(0).to(device)
                with torch.no_grad():
                    e = model(x)[0].float().cpu().numpy()
                any_present = 1.0
            except Exception as ex:
                print(f"  [warn] {pid} {typ}: {str(ex)[:70]}", flush=True)
        embs.append(e)
    return np.concatenate(embs).astype(np.float32), any_present


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="N subjects/split (debug)")
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument(
        "--dry_run", action="store_true", help="encode but do not write pickles"
    )
    args = ap.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Compute nodes have NO internet: load DINOv2 fully offline from the shared
    # local cache (repo + weights copied under cair-clean/.dinov2_cache/).
    HUB = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".dinov2_cache")
    )
    repo = os.path.join(HUB, "facebookresearch_dinov2_main")
    weights = os.path.join(HUB, "checkpoints", "dinov2_vits14_pretrain.pth")
    model = torch.hub.load(
        repo, "dinov2_vits14", source="local", pretrained=False, verbose=False
    )
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model = model.eval().to(device)
    age_map = load_clinical()
    ecg_map = load_ecg_scalars()
    print(
        f"clinical pids={len(age_map)} ecg pids={len(ecg_map)} device={device}",
        flush=True,
    )

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    cache = {}
    age_acc, ecg_acc = [], []  # train stats
    ret_train = []  # train retinal embeddings for PCA
    for sp in splits:
        recs = pickle.load(open(f"{MM}/aireadi_cgm_{sp}.pkl", "rb"))
        if args.limit:
            recs = recs[: args.limit]
        cache[sp] = recs
        t0 = time.time()
        for i, r in enumerate(recs):
            pid = str(r["person_id"])
            age = age_map.get(pid, np.nan)
            label = int(r.get("label", 0))
            ecg = ecg_map.get(pid, np.full(8, np.nan, np.float32))
            ret, ret_p = encode_retinal(pid, model, device)
            r["_static_raw"] = (age, label, ecg, ret, ret_p)
            if sp == "train":
                if np.isfinite(age):
                    age_acc.append(age)
                if np.isfinite(ecg).any():
                    ecg_acc.append(ecg)
                if ret_p > 0:
                    ret_train.append(ret)
            if (i + 1) % 100 == 0:
                print(f"  {sp}: {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{sp}] static encoded ({time.time()-t0:.0f}s)", flush=True)

    if args.dry_run:
        r0 = cache[splits[0]][0]
        age, label, ecg, ret, ret_p = r0["_static_raw"]
        print(
            f"[dry_run] sample: age={age} label={label} ecg(8)={np.round(ecg,1)} "
            f"ret(1536) present={ret_p} ret_norm={np.linalg.norm(ret):.2f}"
        )
        print(f"[dry_run] retinal-present count (train acc)={len(ret_train)}")
        return

    # stats
    age_mean, age_std = float(np.mean(age_acc)), float(np.std(age_acc) or 1.0)
    ecg_arr = np.array(ecg_acc, np.float32)
    ecg_mean = np.nanmean(ecg_arr, axis=0)
    ecg_std = np.nanstd(ecg_arr, axis=0)
    ecg_std[ecg_std == 0] = 1.0
    # PCA on train retinal (1536 -> 128)
    RM = np.array(ret_train, np.float32)
    ret_mu = RM.mean(0)
    U, S, Vt = np.linalg.svd(RM - ret_mu, full_matrices=False)
    comps = Vt[:128]  # (128,1536)
    print(
        f"[stats] age {age_mean:.1f}+-{age_std:.1f} | PCA retinal var kept "
        f"{float((S[:128]**2).sum()/ (S**2).sum()):.2f}",
        flush=True,
    )

    def lbl_onehot(label):
        v = np.zeros(4, np.float32)
        v[min(max(label, 0), 3)] = 1.0
        return v

    for sp in splits:
        for r in cache[sp]:
            age, label, ecg, ret, ret_p = r.pop("_static_raw")
            # clinical(8): [age_z, label_onehot(4), 0,0,0]
            az = (age - age_mean) / age_std if np.isfinite(age) else 0.0
            clinical = np.concatenate(
                [[az], lbl_onehot(label), np.zeros(3, np.float32)]
            ).astype(np.float32)
            r["mm_clinical"] = clinical
            r["mm_clinical_p"] = np.float32(1.0 if np.isfinite(age) else 0.0)
            # ecg(8): z-norm scalars, zero where missing
            ez = (ecg - ecg_mean) / ecg_std
            ez[~np.isfinite(ez)] = 0.0
            r["mm_ecg_emb"] = ez.astype(np.float32)
            r["mm_ecg_emb_p"] = np.float32(1.0 if np.isfinite(ecg).any() else 0.0)
            # retinal(128): PCA-projected
            rp = ((ret - ret_mu) @ comps.T).astype(np.float32)
            r["mm_retinal_emb"] = rp
            r["mm_retinal_emb_p"] = np.float32(ret_p)
        outp = f"{MM}/aireadi_cgm_{sp}.pkl"
        tmp = outp + ".tmp"
        pickle.dump(cache[sp], open(tmp, "wb"))
        os.replace(tmp, outp)
        print(f"[{sp}] static written -> {outp}", flush=True)

    nsp = f"{MM}/mm_norm_stats.json"
    cur = json.load(open(nsp)) if os.path.exists(nsp) else {}
    cur["static"] = {
        "age_mean": age_mean,
        "age_std": age_std,
        "ecg_mean": ecg_mean.tolist(),
        "ecg_std": ecg_std.tolist(),
    }
    json.dump(cur, open(nsp, "w"), indent=2)
    print(f"updated {nsp}", flush=True)


if __name__ == "__main__":
    main()
