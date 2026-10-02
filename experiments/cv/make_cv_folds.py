#!/usr/bin/env python3
"""Generate subject-level K-fold cross-validation splits for a prepared dataset."""

from __future__ import annotations
import argparse, os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cgm_datasets import load_dataset_splits
from cgm_datasets.common import save_prepared_dataset


def subject_of(rec):
    return str(rec.get("person_id", rec.get("subject_id", id(rec))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--data_dir", default=None, help="source prepared dir (defaults to registry)"
    )
    ap.add_argument("--n_folds", type=int, required=True)
    ap.add_argument(
        "--val_holdout",
        type=int,
        default=1,
        help="subjects moved from train to val per fold (EMA selection)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_root",
        default=None,
        help="fold dirs written to <out_root>/fold<k> (default data/cv/<dataset>)",
    )
    args = ap.parse_args()

    tr, va, te, md = load_dataset_splits(args.dataset, prepared_dir=args.data_dir)
    records = list(tr) + list(va) + list(te)
    # group records by subject (a subject may own >1 record; keep them together)
    by_subj = {}
    for r in records:
        by_subj.setdefault(subject_of(r), []).append(r)
    subjects = sorted(by_subj)
    # deterministic shuffle
    import random

    rng = random.Random(args.seed)
    rng.shuffle(subjects)
    K = args.n_folds
    if K > len(subjects):
        raise SystemExit(f"n_folds={K} > n_subjects={len(subjects)} for {args.dataset}")
    folds = [subjects[i::K] for i in range(K)]  # round-robin assignment

    out_root = args.out_root or os.path.join(ROOT, "data", "cv", args.dataset)
    manifest = {
        "dataset": args.dataset,
        "n_folds": K,
        "n_subjects": len(subjects),
        "val_holdout": args.val_holdout,
        "seed": args.seed,
        "folds": [],
    }
    for k in range(K):
        test_subj = set(folds[k])
        remainder = [s for s in subjects if s not in test_subj]
        val_subj = set(remainder[: args.val_holdout])
        train_subj = [s for s in remainder if s not in val_subj]

        def collect(subs):
            out = []
            for s in subs:
                out.extend(by_subj[s])
            return out

        train_r = collect(train_subj)
        val_r = collect(val_subj)
        test_r = collect(sorted(test_subj))
        fold_dir = os.path.join(out_root, f"fold{k}")
        save_prepared_dataset(
            fold_dir, args.dataset, {"train": train_r, "val": val_r, "test": test_r}, md
        )
        manifest["folds"].append(
            {
                "fold": k,
                "dir": fold_dir,
                "n_train_subj": len(train_subj),
                "n_val_subj": len(val_subj),
                "n_test_subj": len(test_subj),
                "n_train_rec": len(train_r),
                "n_val_rec": len(val_r),
                "n_test_rec": len(test_r),
                "test_subjects": sorted(test_subj),
            }
        )
        print(
            f"  fold{k}: train {len(train_subj)}subj/{len(train_r)}rec  "
            f"val {len(val_subj)}/{len(val_r)}  test {len(test_subj)}/{len(test_r)}"
        )

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "cv_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {K} folds -> {out_root}")


if __name__ == "__main__":
    main()
