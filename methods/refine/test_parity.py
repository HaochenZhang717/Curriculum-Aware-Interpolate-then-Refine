"""Check that zero-valued additional modalities preserve checkpoint predictions."""

from utils.paths import DATA_ROOT, REPO_ROOT, CACHE_ROOT
import numpy as np
import torch

from methods.refine.refine import _load_member, Refine
from methods.refine._cgm_mae_core import warm_start_from

D = str(DATA_ROOT)
CKDIR = str(REPO_ROOT / "method_checkpoints/refine")
MEMBERS = [
    "member0_hpo3_a07_l4.pt",
    "member1_hpo4_seed1.pt",
    "member2_hpo4_seed2.pt",
    "member3_hpo4_seed3.pt",
    "member4_hpo4_seed4.pt",
]


def _build_extended(ckpt_path, n_extra_cond, ctx_dim):
    """Rebuild a member's architecture with K extra cond cols + S static ctx."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck.get("config", {})
    return Refine(
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
        n_extra_cond=n_extra_cond,
        ctx_dim=ctx_dim,
    )


def _parity_one(member, K, S, atol=1e-5):
    ckpt = f"{CKDIR}/{member}"
    stock = _load_member(ckpt, "cpu")
    ext = _build_extended(ckpt, K, S)
    warm_start_from(ext, ckpt)
    ext.eval().to("cpu")
    rng = np.random.default_rng(0)
    ts = rng.standard_normal(700).astype(np.float32)
    mask = (rng.random(700) > 0.25).astype(np.float32)
    y_stock = stock.impute(ts, mask, device="cpu")
    y_ext = ext.impute(
        ts,
        mask,
        device="cpu",
        modality=np.zeros((700, K), np.float32) if K else None,
        ctx_vec=np.zeros(S, np.float32) if S else None,
    )
    maxdiff = float(np.abs(y_stock - y_ext).max())
    assert np.allclose(y_stock, y_ext, atol=atol), f"{member}: max|diff|={maxdiff}"
    return maxdiff


def test_parity_all_members_ts_only():
    for m in MEMBERS:
        d = _parity_one(m, K=5, S=0)
        print(f"PARITY ts-only {m}: max|diff|={d:.2e}")


def test_parity_all_members_ts_and_ctx():
    for m in MEMBERS:
        d = _parity_one(m, K=12, S=73)
        print(f"PARITY ts+ctx {m}: max|diff|={d:.2e}")


if __name__ == "__main__":
    test_parity_all_members_ts_only()
    test_parity_all_members_ts_and_ctx()
    print("\nPARITY GATE PASSED for all 5 members (ts-only and ts+ctx).")
