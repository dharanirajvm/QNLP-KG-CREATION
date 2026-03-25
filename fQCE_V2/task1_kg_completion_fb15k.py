#!/usr/bin/env python3
"""Task 1 for FB15k-237: knowledge graph completion using the strong FB15k snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODULE1_DIR = THIS_DIR.parent / "fQCE" / "module1_kge"
if str(MODULE1_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE1_DIR))

from task1_kg_completion import print_single, run_single  # type: ignore
from downstream_utils import load_context  # type: ignore


DEFAULT_SNAPSHOT_DIR = THIS_DIR.parent / "fQCE" / "inference_snapshots" / "quantum_fb15k237_20260308_174529_updated_20260310_193344"
DEFAULT_DATASET_DIR = THIS_DIR.parent / "fQCE" / "datasets" / "fb15k237"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FB15k-237 KG completion using the trained quantum KGE model.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--mode", choices=["tail", "head", "relation"], default="tail")
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--relation", type=str, default="")
    parser.add_argument("--tail", type=str, default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-known", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = load_context(snapshot_dir=args.snapshot_dir, dataset_dir=args.dataset_dir)
    result = run_single(
        ctx=ctx,
        mode=args.mode,
        head=args.head,
        relation=args.relation,
        tail=args.tail,
        top_k=args.top_k,
        include_known=args.include_known,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_single(result)


if __name__ == "__main__":
    main()
