"""Scan the multimodal test split to find FIVE good example gaps per strategy (5 strategies x 5 examples = 25 panels), then run every completed REFINE ladder rung (every prefix with exactly 5 ensemble checkpoints) on each gap and cache the imputed curves + per-gap RMSEs to a .npz.."""

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT
import os, glob, pickle, json
import numpy as np
import torch

from methods.refine import RefineImputer
from cgm_datasets.multimodal.modality_spec import build_mod_and_ctx, rung_active

from utils.physiological_masking import create_physiological_mask

CGM_MEAN, CGM_STD = 132.05, 42.33
T_LO, T_HI = 288, 576
TEST_PKL = str(DATA_ROOT / "aireadi_cgm_mm/aireadi_cgm_test.pkl")

# Ladder prefixes in order with their cumulative conditioning-modality count.
# We include EVERY prefix that currently has exactly 5 ensemble checkpoints.
RUNG_ORDER = [
    ("r0_control", 0, 0),
    ("r1_hr", 1, 1),
    ("r2_steps", 2, 3),
    ("r3_sleep", 3, 4),
    ("r4_resp", 4, 6),
    ("r5_env", 5, 7),
    ("r6_clinical", 6, 8),
    ("r7_ecg", 7, 9),
    ("r8_retinal", 8, 10),
]

STRATEGIES = ["sleep", "ascending", "meal_post", "dipping", "combined"]
SEEDS = list(range(8))  # deterministic seed sweep per record/strategy
N_PER_STRAT = 5  # examples (distinct participants) per strategy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def available_rungs():
    """Return [(prefix, rung_idx, cond_count), ...] for prefixes with 5 ckpts."""
    out = []
    for prefix, rung, cc in RUNG_ORDER:
        n = len(glob.glob(f"method_checkpoints/refine/mm/{prefix}_m*_seed*.pt"))
        if n == 5:
            out.append((prefix, rung, cc))
        else:
            print(f"  SKIP {prefix}: only {n} checkpoints (need 5)")
    return out


def load_rung(prefix, device):
    paths = sorted(glob.glob(f"method_checkpoints/refine/mm/{prefix}_m*_seed*.pt"))
    assert len(paths) == 5, f"{prefix}: {len(paths)} ckpts"
    imp = RefineImputer(ckpt_paths=paths, device=device)
    cfg = torch.load(paths[0], map_location="cpu", weights_only=False)["config"]
    return imp, cfg.get("ts_mods", []) or [], cfg.get("static_blocks", []) or []


def carve(rec, strat, seed):
    d2 = rec["irg_ts"][T_LO:T_HI].copy()
    m2 = rec["irg_ts_mask"][T_LO:T_HI].copy()
    em, tm = create_physiological_mask(
        d2, m2, strategy=strat, target_ratio=0.20, seed=seed
    )
    obs_d2 = em[:, 0].astype(bool)
    tgt_d2 = tm[:, 0].astype(bool)
    return obs_d2, tgt_d2


def longest_run(mask):
    """Return (start, end) inclusive indices of the longest contiguous True run
    in a 1-D boolean mask, or None if empty."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    best = (idx[0], idx[0])
    cur_s = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            if prev - cur_s > best[1] - best[0]:
                best = (cur_s, prev)
            cur_s = prev = k
    if prev - cur_s > best[1] - best[0]:
        best = (cur_s, prev)
    return int(best[0]), int(best[1])


def gap_quality(rec, tgt_d2):
    """Score a carved gap by its LONGEST CONTIGUOUS run (the run that will be
    zoomed into). Require >=6 contiguous masked samples and a real but
    physiological glucose excursion across that run, with no sensor-artifact
    single-step jumps."""
    if int(tgt_d2.sum()) < 6:
        return None
    run = longest_run(tgt_d2)
    if run is None:
        return None
    g0, g1 = run
    run_len = g1 - g0 + 1
    if run_len < 6:
        return None
    z = rec["irg_ts"][T_LO:T_HI, 0]
    truth_run = z[g0 : g1 + 1] * CGM_STD + CGM_MEAN
    excursion = float(truth_run.max() - truth_run.min())
    max_step = float(np.abs(np.diff(truth_run)).max()) if len(truth_run) > 1 else 0.0
    in_range = bool((truth_run.min() > 30) and (truth_run.max() < 420))
    return dict(
        n=run_len,
        span=run_len,
        contig=1.0,
        excursion=excursion,
        max_step=max_step,
        in_range=in_range,
        start=g0,
        end=g1,
    )


def collect_candidates():
    recs = pickle.load(open(TEST_PKL, "rb"))
    print(f"loaded {len(recs)} test records; device={DEVICE}")
    candidates = {s: [] for s in STRATEGIES}
    for ri, rec in enumerate(recs):
        if int(np.asarray(rec["irg_ts"]).shape[0]) < T_HI:
            continue
        for s in STRATEGIES:
            for seed in SEEDS:
                obs_d2, tgt_d2 = carve(rec, s, seed)
                q = gap_quality(rec, tgt_d2)
                if q is None:
                    continue
                # zoomed panel only needs context on at least ONE side of the
                # longest contiguous run (sleep gaps touch the nocturnal window
                # edge); the plot clamps the zoom window accordingly.
                has_context = q["start"] >= 1 or q["end"] <= (T_HI - T_LO - 2)
                if (
                    has_context
                    and 20.0 <= q["excursion"] <= 200.0
                    and q["max_step"] <= 35.0
                    and q["in_range"]
                    and q["n"] >= 6
                ):
                    q.update(
                        rec_idx=ri,
                        person_id=int(rec["person_id"]),
                        strategy=s,
                        seed=seed,
                    )
                    candidates[s].append(q)
    for s in STRATEGIES:
        print(f"  candidates[{s}] = {len(candidates[s])}")
    return recs, candidates


def choose_five(candidates):
    """Pick 5 DISTINCT participants per strategy (distinct within each row),
    deterministically ranked by a strategy-appropriate quality key. To keep the
    five examples varied we pick across a spread of gap sizes rather than the top
    five near-identical gaps."""
    # rank key per strategy: prefer clearly excursive, decent-length gaps
    keyfn = {
        "sleep": lambda q: (q["n"], q["excursion"]),
        "ascending": lambda q: (q["excursion"], q["n"]),
        "meal_post": lambda q: (q["excursion"], q["n"]),
        "dipping": lambda q: (q["excursion"], q["n"]),
        "combined": lambda q: (q["excursion"], q["n"]),
    }
    chosen = {}
    for s in STRATEGIES:
        # best gap per participant (dedupe within strategy)
        per_pid = {}
        for c in sorted(candidates[s], key=keyfn[s], reverse=True):
            per_pid.setdefault(c["person_id"], c)
        ranked = sorted(per_pid.values(), key=keyfn[s], reverse=True)
        if len(ranked) < N_PER_STRAT:
            raise RuntimeError(
                f"strategy {s!r}: only {len(ranked)} distinct participants"
            )
        # spread picks evenly across the ranked list for visual variety
        n = len(ranked)
        sel_idx = sorted(
            {int(round(j * (n - 1) / (N_PER_STRAT - 1))) for j in range(N_PER_STRAT)}
        )
        # if rounding collapsed duplicates (small n), backfill from the top
        k = 0
        while len(sel_idx) < N_PER_STRAT:
            if k not in sel_idx:
                sel_idx.append(k)
            k += 1
        sel_idx = sorted(sel_idx)[:N_PER_STRAT]
        picks = [ranked[j] for j in sel_idx]
        chosen[s] = picks
        for c in picks:
            print(
                f"[{s:>10s}] pid={c['person_id']} idx={c['rec_idx']} "
                f"seed={c['seed']} run_len={c['n']} "
                f"excursion={c['excursion']:.1f}"
            )
    return chosen


def main():
    rungs = available_rungs()
    print("rungs used:", [p for p, _, _ in rungs])
    recs, candidates = collect_candidates()
    chosen = choose_five(candidates)

    # flatten to 25 panels in row-major (strategy-major) order. The panel's gap
    # (used for shading, zoom, ground-truth curve, and in-gap RMSE) is the
    # LONGEST CONTIGUOUS run of the carved target - i.e. the region we zoom into.
    panels = []
    for s in STRATEGIES:
        for c in chosen[s]:
            rec = recs[c["rec_idx"]]
            obs_d2, tgt_full = carve(rec, s, c["seed"])
            g0, g1 = c["start"], c["end"]
            tgt_run = np.zeros_like(tgt_full)
            tgt_run[g0 : g1 + 1] = True
            full_ts = rec["irg_ts"][:, 0].astype(np.float32)
            truth_z = full_ts[T_LO:T_HI]
            panels.append(
                dict(
                    strategy=s,
                    person_id=c["person_id"],
                    rec_idx=c["rec_idx"],
                    seed=c["seed"],
                    obs_d2=obs_d2,
                    tgt_d2=tgt_run,
                    truth_z=truth_z,
                    gap_start=g0,
                    gap_end=g1,
                    preds={},
                    rmse={},
                )
            )

    # run every rung's ensemble on each gap
    for prefix, rung, cc in rungs:
        imp, ts_mods, static_blocks = load_rung(prefix, DEVICE)
        exp_ts, exp_ctx = rung_active(rung)
        print(f"{prefix}: ts_mods={ts_mods} (spec {exp_ts}) static={static_blocks}")
        for panel in panels:
            rec = recs[panel["rec_idx"]]
            mod, ctx = build_mod_and_ctx(rec, ts_mods, static_blocks)
            full_obs = rec["irg_ts_mask"][:, 0].astype(bool).copy()
            full_obs[T_LO:T_HI] = panel["obs_d2"]
            full_ts = rec["irg_ts"][:, 0].astype(np.float32)
            pred_z = imp.impute(
                full_ts, full_obs.astype(np.float32), modality=mod, ctx_vec=ctx
            )
            pred_d2_z = pred_z[T_LO:T_HI]
            tgt = panel["tgt_d2"]
            pred_mgdl = pred_d2_z[tgt] * CGM_STD + CGM_MEAN
            truth_mgdl = panel["truth_z"][tgt] * CGM_STD + CGM_MEAN
            rmse = float(np.sqrt(np.mean((pred_mgdl - truth_mgdl) ** 2)))
            panel["preds"][prefix] = pred_d2_z.astype(np.float32)
            panel["rmse"][prefix] = rmse
        rmses = [panel["rmse"][prefix] for panel in panels]
        print(
            f"  {prefix}: mean in-gap RMSE over 25 panels = "
            f"{np.mean(rmses):6.2f} mg/dL"
        )
        del imp
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # cache to npz
    out = {}
    meta = []
    prefixes = [p for p, _, _ in rungs]
    for i, panel in enumerate(panels):
        out[f"p{i}_obs"] = panel["obs_d2"]
        out[f"p{i}_tgt"] = panel["tgt_d2"]
        out[f"p{i}_truth_z"] = panel["truth_z"]
        for prefix in prefixes:
            out[f"p{i}_pred_{prefix}"] = panel["preds"][prefix]
        meta.append(
            dict(
                idx=i,
                strategy=panel["strategy"],
                person_id=panel["person_id"],
                rec_idx=panel["rec_idx"],
                seed=panel["seed"],
                gap_start=panel["gap_start"],
                gap_end=panel["gap_end"],
                rmse={p: panel["rmse"][p] for p in prefixes},
            )
        )
    out["meta_json"] = json.dumps(meta)
    out["rungs_json"] = json.dumps([[p, r, c] for p, r, c in rungs])
    cache = "experiments/evaluation/_imputation_examples_cache.npz"
    np.savez(cache, **out)
    print(f"\nwrote cache -> {cache}")
    print(
        json.dumps(
            [
                {k: m[k] for k in ("strategy", "person_id", "gap_start", "gap_end")}
                for m in meta
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
