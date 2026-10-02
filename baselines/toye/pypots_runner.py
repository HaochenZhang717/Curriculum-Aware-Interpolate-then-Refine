#!/usr/bin/env python3
"""Standalone PyPOTS baseline runner (executed by the PyPOTS env python)."""

import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument(
        "--method", choices=["mrnn", "gpvae", "saits", "brits"], required=True
    )
    # Default epochs may be overridden via the PYPOTS_EPOCHS env var so callers
    # that reuse the in-proc _run_pypots helper (which does not forward -epochs)
    # can still control training length. An explicit -epochs flag wins.
    ap.add_argument(
        "--epochs", type=int, default=int(os.environ.get("PYPOTS_EPOCHS", "100"))
    )
    args = ap.parse_args()

    data = np.load(args.inp)
    X = data["X"].astype(np.float32)  # (N, L, 1), NaN at missing
    N, L, K = X.shape

    if args.method == "mrnn":
        from pypots.imputation import MRNN

        model = MRNN(n_steps=L, n_features=K, rnn_hidden_size=64, epochs=args.epochs)
    elif args.method == "gpvae":
        from pypots.imputation import GPVAE

        model = GPVAE(n_steps=L, n_features=K, latent_size=8, epochs=args.epochs)
    elif args.method == "saits":
        # Standard SAITS benchmark config (2 attention layers, d_model=256).
        from pypots.imputation import SAITS

        model = SAITS(
            n_steps=L,
            n_features=K,
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=128,
            dropout=0.1,
            batch_size=32,
            epochs=args.epochs,
        )
    else:  # brits
        from pypots.imputation import BRITS

        model = BRITS(
            n_steps=L,
            n_features=K,
            rnn_hidden_size=64,
            batch_size=32,
            epochs=args.epochs,
        )

    model.fit({"X": X})
    imputed = np.asarray(model.impute({"X": X}), dtype=np.float32)
    # GP-VAE is probabilistic and returns (N, n_samples, L, K); average the
    # sampling axis to a single (N, L, K) imputation. MRNN is already (N, L, K).
    if imputed.ndim == 4:
        imputed = imputed.mean(axis=1)
    np.savez(args.out, X_imputed=imputed.astype(np.float32))


if __name__ == "__main__":
    main()
