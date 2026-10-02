"""Train CAIR or evaluate a published missingness protocol."""

import argparse
from pathlib import Path

from utils.paths import DATA_ROOT, REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description="CAIR")
    parser.add_argument(
        "-p",
        choices=["train", "mechanisms", "shortgap", "physiological"],
        default="mechanisms",
        help="training or evaluation protocol",
    )
    parser.add_argument(
        "-d", dest="dataset", default="aireadi", help="dataset registry name"
    )
    parser.add_argument(
        "-r", dest="data_dir", default=None, help="prepared data directory"
    )
    parser.add_argument(
        "-c",
        dest="ckpt_dir",
        default=str(REPO_ROOT / "method_checkpoints/refine"),
        help="checkpoint directory",
    )
    parser.add_argument(
        "-g", dest="gpu", type=int, default=0, help="GPU index; -1 uses CPU"
    )
    parser.add_argument(
        "-n",
        dest="n_participants",
        type=int,
        default=352,
        help="evaluation participants",
    )
    parser.add_argument(
        "-m",
        dest="method",
        default="refine",
        choices=["refine", "linear", "pchip", "akima"],
        help="short-gap imputer",
    )
    parser.add_argument(
        "-o", dest="out", default=None, help="result or training checkpoint path"
    )
    parser.add_argument(
        "-e", dest="epochs", type=int, default=100, help="training epochs"
    )
    parser.add_argument("-s", dest="seed", type=int, default=1, help="training seed")
    parser.add_argument(
        "-b", dest="batch_size", type=int, default=64, help="training batch size"
    )
    parser.add_argument(
        "-a",
        dest="ts_mods",
        default="",
        help="conditioning channels, separated by commas",
    )
    parser.add_argument(
        "-x", dest="pypots_python", default=None, help="PyPOTS environment interpreter"
    )
    args = parser.parse_args()
    if args.method != "refine" and args.p != "shortgap":
        parser.error("-m applies to the shortgap protocol")
    if args.p == "physiological" and args.dataset != "aireadi":
        parser.error("physiological masks are defined for AI-READI CGM")
    if args.out is None:
        args.out = str(
            REPO_ROOT
            / (
                f"method_checkpoints/refine/member_seed{args.seed}.pt"
                if args.p == "train"
                else f"results/{args.dataset}_{args.p}_{args.method}.json"
            )
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.p == "train":
        from methods.refine.train_refine import train

        args.datasets = args.dataset
        args.stage = "finetune"
        args.mix = "equal"
        args.init_from = None
        args.window = 576
        args.lr = 3e-4
        args.interp_hidden = 128
        args.interp_layers = 4
        args.inject = "residual"
        args.interp_type = "gru"
        args.interp_in = "basic"
        args.aux = 0.7
        args.aux_huber = False
        args.static_blocks = ""
        args.mask_mode = "realistic"
        train(args)
    elif args.p == "mechanisms":
        args.refine_ckpt_glob = str(Path(args.ckpt_dir) / "*.pt")
        args.n_refine_members = 5
        args.skip_refine = False
        args.skip_baselines = False
        args.pypots_epochs = 100
        args.mm_pkl = str(DATA_ROOT / "aireadi_cgm_mm/aireadi_cgm_test.pkl")
        if args.dataset.startswith("mimic"):
            from experiments.evaluation.eval_toye_mimic import run
        else:
            if not Path(args.mm_pkl).is_file():
                parser.error(
                    f"MAR evaluation requires activity channels at {args.mm_pkl}"
                )
            from experiments.evaluation.eval_toye_benchmark import run
        run(args)
    else:
        args.ckpt = None
        args.ckpt_paths = None
        args.n_refinements = 2
        if args.p == "shortgap":
            from experiments.evaluation.eval_short_gaps import main as evaluate

            args.all_splits = False
            args.gaps = "3,6,9,12"
            args.stride = 1
            args.n_placements = 10
            args.impute_max_len = 0
        else:
            from experiments.evaluation.eval_refine import main as evaluate

            args.n_masks = 5
        evaluate(args)


if __name__ == "__main__":
    main()
