#!/usr/bin/env python3
"""Build paper-ready tables from the Toye benchmark JSON."""

from __future__ import annotations
import json


def build_metric_table(json_path, metric, mech, group=None):
    """Return {(method, pct): value} for one metric and mechanism."""
    if group is not None:
        return build_metric_table_by_group(json_path, metric, mech, group)
    d = json.load(open(json_path))
    table = {}
    for key, vals in d["aggregate"].items():
        method, m, pct = key.split("|")
        if m != mech:
            continue
        if metric in vals:
            table[(method, float(pct))] = vals[metric]
    return table


def build_metric_table_by_group(json_path, metric, mech, group):
    import collections

    d = json.load(open(json_path))
    acc = collections.defaultdict(list)
    for c in d["cells"]:
        if (
            c.get("mech") == mech
            and c.get("group") == group
            and metric in c
            and c[metric] == c[metric]
        ):
            acc[(c["method"], c["pct"])].append(c[metric])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def print_table(json_path, metric, mech):
    tbl = build_metric_table(json_path, metric, mech)
    methods = sorted({k[0] for k in tbl})
    pcts = sorted({k[1] for k in tbl})
    header = "method".ljust(12) + "".join(f"{int(p*100):>8}" for p in pcts)
    print(f"# {metric} under {mech}")
    print(header)
    for meth in methods:
        row = meth.ljust(12) + "".join(
            f"{tbl.get((meth, p), float('nan')):8.3f}" for p in pcts
        )
        print(row)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--metric", default="mrr_tir")
    ap.add_argument("--mech", default="nmar")
    a = ap.parse_args()
    print_table(a.json_path, a.metric, a.mech)
