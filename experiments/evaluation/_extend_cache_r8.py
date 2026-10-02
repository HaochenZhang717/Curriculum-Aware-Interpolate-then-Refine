"""Extend the imputation-examples cache with the r8_retinal rung WITHOUT re-selecting examples or re-running the other rungs."""

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT
import os, glob, pickle, json
import numpy as np
import torch

from methods.refine import RefineImputer
from cgm_datasets.multimodal.modality_spec import build_mod_and_ctx, rung_active

CGM_MEAN, CGM_STD = 132.05, 42.33
T_LO, T_HI = 288, 576
TEST_PKL = str(DATA_ROOT / "aireadi_cgm_mm/aireadi_cgm_test.pkl")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_imputation_examples_cache.npz")

NEW_RUNG = ("r8_retinal", 8, 10)  # (prefix, rung_idx, cumulative cond-count)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_rung(prefix, device):
    paths = sorted(glob.glob(f"method_checkpoints/refine/mm/{prefix}_m*_seed*.pt"))
    assert len(paths) == 5, f"{prefix}: {len(paths)} ckpts"
    imp = RefineImputer(ckpt_paths=paths, device=device)
    cfg = torch.load(paths[0], map_location="cpu", weights_only=False)["config"]
    return imp, cfg.get("ts_mods", []) or [], cfg.get("static_blocks", []) or []


def main():
    prefix, rung, cc = NEW_RUNG
    d = np.load(CACHE, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    rungs = [list(r) for r in json.loads(str(d["rungs_json"]))]
    if any(r[0] == prefix for r in rungs):
        print(f"{prefix} already in cache; nothing to do.")
        return

    # keep every existing array
    store = {k: d[k] for k in d.files if k not in ("meta_json", "rungs_json")}

    recs = pickle.load(open(TEST_PKL, "rb"))
    print(f"loaded {len(recs)} test records; device={DEVICE}")

    imp, ts_mods, static_blocks = load_rung(prefix, DEVICE)
    exp_ts, exp_ctx = rung_active(rung)
    print(
        f"{prefix}: ts_mods={ts_mods} (spec {exp_ts}) static={static_blocks} "
        f"(spec {exp_ctx})"
    )

    rmses = []
    for m in meta:
        i = m["idx"]
        rec = recs[m["rec_idx"]]
        obs_d2 = d[f"p{i}_obs"].astype(bool)
        tgt = d[f"p{i}_tgt"].astype(bool)
        truth_z = d[f"p{i}_truth_z"]

        mod, ctx = build_mod_and_ctx(rec, ts_mods, static_blocks)
        full_obs = rec["irg_ts_mask"][:, 0].astype(bool).copy()
        full_obs[T_LO:T_HI] = obs_d2
        full_ts = rec["irg_ts"][:, 0].astype(np.float32)
        pred_z = imp.impute(
            full_ts, full_obs.astype(np.float32), modality=mod, ctx_vec=ctx
        )
        pred_d2_z = pred_z[T_LO:T_HI].astype(np.float32)

        pred_mgdl = pred_d2_z[tgt] * CGM_STD + CGM_MEAN
        truth_mgdl = truth_z[tgt] * CGM_STD + CGM_MEAN
        rmse = float(np.sqrt(np.mean((pred_mgdl - truth_mgdl) ** 2)))

        store[f"p{i}_pred_{prefix}"] = pred_d2_z
        m.setdefault("rmse", {})[prefix] = rmse
        rmses.append(rmse)

    print(
        f"  {prefix}: mean in-gap RMSE over {len(meta)} panels = "
        f"{np.mean(rmses):6.2f} mg/dL"
    )
    del imp
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    rungs.append([prefix, rung, cc])
    store["meta_json"] = json.dumps(meta)
    store["rungs_json"] = json.dumps(rungs)
    np.savez(CACHE, **store)
    print(f"extended cache -> {CACHE}")
    print("rungs now:", [r[0] for r in rungs])


if __name__ == "__main__":
    main()
