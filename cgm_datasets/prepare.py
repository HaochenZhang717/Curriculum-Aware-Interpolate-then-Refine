from __future__ import annotations

import argparse
import json
import os

from cgm_datasets import list_datasets, load_dataset_splits, prepare_dataset


def cmd_prepare(args: argparse.Namespace) -> None:
    output_dir = prepare_dataset(
        args.dataset,
        args.source,
        output_dir=args.output_dir,
        force=args.force,
    )
    _, _, _, metadata = load_dataset_splits(args.dataset, prepared_dir=output_dir)
    print(
        json.dumps({"output_dir": output_dir, "metadata": metadata.to_dict()}, indent=2)
    )


def cmd_inspect(args: argparse.Namespace) -> None:
    train, val, test, metadata = load_dataset_splits(
        args.dataset, prepared_dir=args.data_dir
    )
    payload = {
        "dataset": metadata.to_dict(),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
    }
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and inspect CGM datasets.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare", help="Convert a raw dataset into the prepared split format."
    )
    prepare.add_argument("--dataset", required=True, choices=list_datasets())
    prepare.add_argument(
        "--source", required=True, help="Raw dataset file or directory."
    )
    prepare.add_argument(
        "--output_dir", default=None, help="Prepared dataset directory."
    )
    prepare.add_argument(
        "--force", action="store_true", help="Overwrite an existing prepared dataset."
    )
    prepare.set_defaults(func=cmd_prepare)

    inspect = sub.add_parser("inspect", help="Inspect a prepared dataset.")
    inspect.add_argument("--dataset", default="aireadi", choices=list_datasets())
    inspect.add_argument(
        "--data_dir", default=None, help="Prepared dataset directory override."
    )
    inspect.set_defaults(func=cmd_inspect)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
