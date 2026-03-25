#!/usr/bin/env python3
"""Task 2 for FB15k-237: semantic similarity over the trained KGE entity space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODULE1_DIR = THIS_DIR.parent / "fQCE" / "module1_kge"
if str(MODULE1_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE1_DIR))

from downstream_utils import load_context  # type: ignore
from task2_semantic_retrieval import run_semantic_similarity  # type: ignore


DEFAULT_SNAPSHOT_DIR = THIS_DIR.parent / "fQCE" / "inference_snapshots" / "quantum_fb15k237_20260308_174529_updated_20260310_193344"
DEFAULT_DATASET_DIR = THIS_DIR.parent / "fQCE" / "datasets" / "fb15k237"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic similarity over FB15k-237 entity embeddings.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--anchor", type=str, required=True)
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = load_context(snapshot_dir=args.snapshot_dir, dataset_dir=args.dataset_dir)
    result = run_semantic_similarity(
        ctx=ctx,
        anchor=args.anchor,
        target=args.target,
        top_k=args.top_k,
        include_self=args.include_self,
    )
    result["task"] = "semantic_similarity_fb15k237"
    result["snapshot_dir"] = str(args.snapshot_dir.resolve())
    result["dataset_dir"] = str(args.dataset_dir.resolve())

    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Anchor:", result["anchor"])
    if "target" in result:
        print("Target:", result["target"])
        print(f"Cosine similarity: {result['cosine_similarity']:.6f}")
    print("Nearest neighbors:")
    for row in result["neighbors"]:
        print(f" {row['rank']:>2}. {row['neighbor']} | cosine={row['cosine_similarity']:.6f}")


if __name__ == "__main__":
    main()
