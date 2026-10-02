#!/usr/bin/env python3
"""Build a derived AI-READI prepared split where a wearable channel (hr|resp) becomes the imputation TARGET (irg_ts / irg_ts_mask), for CAIR + baseline short-gap benchmarking."""

from __future__ import annotations
import argparse, json, os, pickle
import numpy as np

# target name -> (value key, presence key) in the source MM split
MM_KEY = {"hr": ("mm_hr", "mm_hr_p"), "resp": ("mm_resp", "mm_resp_p")}


def build_records(records, target):
    vkey, pkey = MM_KEY[target]
    out = []
    for r in records:
        p = np.asarray(r[pkey], np.float32).reshape(-1)
        if float(p.sum()) <= 0.0:  # drop 0-coverage: no supervisable points
            continue
        r2 = dict(r)  # shallow copy; keeps mm_* conditioning keys
        r2["irg_ts"] = np.asarray(r[vkey], np.float32).reshape(-1, 1)
        r2["irg_ts_mask"] = np.asarray(r[pkey], np.float32).reshape(-1, 1)
        r2["cgm_target_missing"] = 0
        out.append(r2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm_dir", required=True, help="source aireadi_cgm_mm dir")
    ap.add_argument("--target", required=True, choices=["hr", "resp"])
    ap.add_argument("--out", required=True, help="output prepared-split dir")
    args = ap.parse_args()

    stats = json.load(open(os.path.join(args.mm_dir, "mm_norm_stats.json")))
    st = stats["mm_stats"][args.target]  # {"mean":..., "std":...}
    os.makedirs(args.out, exist_ok=True)

    counts = {}
    for sp in ("train", "val", "test"):
        recs = pickle.load(
            open(os.path.join(args.mm_dir, f"aireadi_cgm_{sp}.pkl"), "rb")
        )
        built = build_records(recs, args.target)
        counts[sp] = {"in": len(recs), "kept": len(built)}
        pickle.dump(built, open(os.path.join(args.out, f"aireadi_cgm_{sp}.pkl"), "wb"))
        print(f"[{sp}] kept {len(built)}/{len(recs)}", flush=True)

    # metadata.json: keys MUST be a subset of DatasetMetadata fields (from_dict does cls(**data)).
    meta = {
        "name": "aireadi",
        "display_name": f"AI-READI {args.target.upper()} (imputation target)",
        "description": f"{args.target} swapped into irg_ts; derived from {args.mm_dir}",
        "sampling_minutes": 5,
        "day_len": 288,
        "eval_window_start": 288,
        "eval_window_end": 576,
        "glucose_mean_mgdl": float(
            st["mean"]
        ),  # native mean (units of the target signal)
        "glucose_std_mgdl": float(st["std"]),  # native std  -> native-unit RMSE in eval
        "glucose_normalization": f"native_{args.target}_zscore",
        "aligned_to_midnight": True,
        "supported_modalities": [args.target],
        "notes": {
            "derived_from": args.mm_dir,
            "target_channel": MM_KEY[args.target][0],
            "native_mean": float(st["mean"]),
            "native_std": float(st["std"]),
        },
    }
    json.dump(meta, open(os.path.join(args.out, "metadata.json"), "w"), indent=2)
    print(f"counts={counts} native_std={st['std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
