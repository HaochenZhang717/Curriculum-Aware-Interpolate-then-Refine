#!/usr/bin/env python3
"""CAIR training with refinement loss, interpolation loss, and EMA."""

from __future__ import annotations
import argparse, copy, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods.cair.backbone import (
    CAIRDataset,
    make_train_loader,
    warm_start_from,
    CGM_STD,
    PASS1_WEIGHT,
    PASS2_WEIGHT,
    PASS3_WEIGHT,
)
from methods.cair.cair import CAIR
from cgm_datasets.pooled import load_pooled_records


class EMA:
    """Exponential moving average of trainable params (decay 0.999)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def copy_to(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    p.data.copy_(self.shadow[n])


def build_model(
    *,
    n_datasets,
    stage,
    interp_hidden,
    interp_layers,
    inject="residual",
    interp_type="gru",
    interp_in="basic",
    d_model=128,
    n_layers=8,
    ff_dim=512,
    n_heads=8,
    window=576,
    n_extra_cond=0,
    ctx_dim=0,
):
    model = CAIR(
        interp_hidden=interp_hidden,
        interp_layers=interp_layers,
        inject=inject,
        interp_type=interp_type,
        interp_in=interp_in,
        d_model=d_model,
        n_layers=n_layers,
        ff_dim=ff_dim,
        n_heads=n_heads,
        window=window,
        n_datasets=n_datasets,
        n_extra_cond=n_extra_cond,
        ctx_dim=ctx_dim,
    )
    if stage == "finetune":
        # AI-READI is domain 0; keep dom_emb at zero so impute()/eval (no dom_idx) is exact.
        model.dom_emb.weight.requires_grad_(False)
    return model


def cair_loss(
    model, obs, msk, pos, tod, cond, gt, tgt, dom, *, aux, aux_huber, ctx=None
):
    """3-pass refinement MSE + auxiliary supervised interpolation loss."""
    p1, p2, p3 = model(
        obs, msk, pos, tod, cond, three_pass=True, dom_idx=dom, ctx_vec=ctx
    )
    tgt_f = tgt.float()
    nt = tgt_f.sum() + 1e-8
    l1 = ((p1 - gt) ** 2 * tgt_f).sum() / nt
    l2 = ((p2 - gt) ** 2 * tgt_f).sum() / nt
    l3 = ((p3 - gt) ** 2 * tgt_f).sum() / nt
    loss = PASS1_WEIGHT * l1 + PASS2_WEIGHT * l2 + PASS3_WEIGHT * l3
    if aux > 0:
        y0 = model.interp_pred(obs, msk.long().squeeze(-1), pos, tod, cond)
        if aux_huber:
            aux_l = (
                F.huber_loss(y0, gt, reduction="none", delta=1.0) * tgt_f
            ).sum() / nt
        else:
            aux_l = ((y0 - gt) ** 2 * tgt_f).sum() / nt
        loss = loss + aux * aux_l
    return loss


def _val_pass3_rmse(model, dl_val, device):
    model.eval()
    tot = 0.0
    cnt = 0
    with torch.no_grad():
        for obs, msk, pos, tod, cond, gt, tgt, dom, ctx in dl_val:
            obs, msk, pos, tod = (
                obs.to(device),
                msk.to(device),
                pos.to(device),
                tod.to(device),
            )
            cond, gt, tgt, dom = (
                cond.to(device),
                gt.to(device),
                tgt.to(device),
                dom.to(device),
            )
            ctx = ctx.to(device)
            _, _, p3 = model(
                obs, msk, pos, tod, cond, three_pass=True, dom_idx=dom, ctx_vec=ctx
            )
            tgt_f = tgt.float()
            tot += float(((p3 - gt) ** 2 * tgt_f).sum())
            cnt += int(tgt.sum())
    return float(np.sqrt(tot / max(cnt, 1)) * CGM_STD)


def train(args):
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    train_r, dom_names = load_pooled_records(
        datasets, split="train", data_dir=args.data_dir
    )
    val_r, _ = load_pooled_records(datasets, split="val", data_dir=args.data_dir)
    n_datasets = len(dom_names)

    from cgm_datasets.multimodal.modality_spec import (
        TS_MODALITIES,
        STATIC_BLOCKS,
        ts_width,
        ctx_width,
        parse_csv,
    )

    ts_mods = [m for m in TS_MODALITIES if m in parse_csv(args.ts_mods)]
    static_blocks = [b for b in STATIC_BLOCKS if b in parse_csv(args.static_blocks)]
    K = ts_width(ts_mods)
    S = ctx_width(static_blocks)
    print(
        f"[train_cair] stage={args.stage} datasets={dom_names} "
        f"train={len(train_r)} val={len(val_r)} device={device} "
        f"ts_mods={ts_mods} static_blocks={static_blocks} K={K} S={S}",
        flush=True,
    )

    train_stride = getattr(args, "stride", 144)
    ds_train = CAIRDataset(
        train_r,
        window=args.window,
        stride=train_stride,
        seed=42,
        ts_mods=ts_mods,
        static_blocks=static_blocks,
        mask_mode=args.mask_mode,
    )
    ds_val = CAIRDataset(
        val_r,
        window=args.window,
        stride=args.window,
        seed=99,
        ts_mods=ts_mods,
        static_blocks=static_blocks,
        mask_mode=args.mask_mode,
    )
    pin = device.type == "cuda"
    if args.stage == "finetune":
        dl_train = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=pin,
            drop_last=True,
        )
    else:
        dl_train = make_train_loader(
            ds_train, batch_size=args.batch_size, mix=args.mix, num_workers=4
        )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin
    )

    model = build_model(
        n_datasets=n_datasets,
        stage=args.stage,
        interp_hidden=args.interp_hidden,
        interp_layers=args.interp_layers,
        inject=args.inject,
        interp_type=args.interp_type,
        interp_in=args.interp_in,
        window=args.window,
        d_model=getattr(args, "d_model", 128),
        n_layers=getattr(args, "n_layers", 8),
        ff_dim=getattr(args, "ff_dim", 512),
        n_heads=getattr(args, "n_heads", 8),
        n_extra_cond=K,
        ctx_dim=S,
    ).to(device)
    if args.init_from and os.path.exists(args.init_from):
        warm_start_from(model, args.init_from)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    ema = EMA(model, decay=0.999)
    ema_eval = copy.deepcopy(model).to(device)
    best = {"rmse": float("inf"), "state": None, "epoch": -1}

    def select(ep):
        ema.copy_to(ema_eval)
        rmse = _val_pass3_rmse(ema_eval, dl_val, device)
        print(f"  ep{ep} val_pass3={rmse:.2f} mg/dL", flush=True)
        if rmse < best["rmse"]:
            best.update(
                rmse=rmse,
                epoch=ep,
                state={
                    k: v.detach().cpu().clone()
                    for k, v in ema_eval.state_dict().items()
                },
            )

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        for obs, msk, pos, tod, cond, gt, tgt, dom, ctx in dl_train:
            obs, msk, pos, tod = (
                obs.to(device),
                msk.to(device),
                pos.to(device),
                tod.to(device),
            )
            cond, gt, tgt, dom = (
                cond.to(device),
                gt.to(device),
                tgt.to(device),
                dom.to(device),
            )
            ctx = ctx.to(device)
            loss = cair_loss(
                model,
                obs,
                msk,
                pos,
                tod,
                cond,
                gt,
                tgt,
                dom,
                aux=args.aux,
                aux_huber=args.aux_huber,
                ctx=ctx,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
        sched.step()
        select(ep)

    cfg = {
        "hidden": args.interp_hidden,
        "layers": args.interp_layers,
        "inject": args.inject,
        "interp_type": args.interp_type,
        "interp_in": args.interp_in,
        "aux": args.aux,
        "aux_huber": args.aux_huber,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "d_model": getattr(args, "d_model", 128),
        "n_layers": getattr(args, "n_layers", 8),
        "ff_dim": getattr(args, "ff_dim", 512),
        "n_heads": getattr(args, "n_heads", 8),
        "window": args.window,
        "n_datasets": n_datasets,
        "n_extra_cond": K,
        "ctx_dim": S,
        "ts_mods": ts_mods,
        "static_blocks": static_blocks,
    }
    state = best["state"] or {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": state, "config": cfg, "sel_epoch": best["epoch"]}, args.out)
    print(
        f"[train_cair] saved {args.out} best_val_pass3={best['rmse']:.2f} "
        f"sel_epoch={best['epoch']} ({time.time()-t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets", default="aireadi", help="comma list, e.g. aireadi,hupa_ucm,ohio"
    )
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--stage", default="pretrain", choices=["pretrain", "finetune"])
    ap.add_argument("--mix", default="equal", choices=["equal", "sqrt", "proportional"])
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--out", required=True, help="output checkpoint path")
    ap.add_argument("--window", type=int, default=576)
    ap.add_argument("--stride", type=int, default=144)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interp_hidden", type=int, default=128)
    ap.add_argument("--interp_layers", type=int, default=4)
    ap.add_argument(
        "--inject", default="residual", choices=["value", "value_slope", "residual"]
    )
    ap.add_argument(
        "--interp_type", default="gru", choices=["gru", "lstm", "tcn", "attn"]
    )
    ap.add_argument("--interp_in", default="basic", choices=["basic", "rich"])
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--ff_dim", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--aux", type=float, default=0.7)
    ap.add_argument("--aux_huber", action="store_true")
    ap.add_argument(
        "--ts_mods", default="", help="csv time-series modalities, e.g. hr,steps,cal"
    )
    ap.add_argument(
        "--static_blocks", default="", help="csv static blocks, e.g. clinical,ecg"
    )
    ap.add_argument(
        "--mask_mode",
        default="physio",
        choices=["physio", "realistic"],
        help="training gap masking: 'physio' (CGM strategies, default/unchanged) "
        "or 'realistic' (generic MCAR+contiguous-gap augmentation for non-CGM signals)",
    )
    train(ap.parse_args())
