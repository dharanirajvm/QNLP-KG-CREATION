#!/usr/bin/env python3
"""Phase 1 explainable reasoning over the strong FB15k quantum KGE snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explanation_utils import (
    DEFAULT_DATASET_DIR,
    DEFAULT_SNAPSHOT_DIR,
    build_explanation_context,
    explain_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a predicted or provided FB15k triple using local paths and embedding support.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--head", type=str, required=True, help="Head entity ID or text, e.g. '/m/02mjmr' or 'Barack Obama'.")
    parser.add_argument("--relation", type=str, required=True, help="Relation ID or text, e.g. '/people/person/place_of_birth' or 'place of birth'.")
    parser.add_argument("--tail", type=str, default="", help="Tail entity ID or text. If omitted, predict the best tail and explain that prediction.")
    parser.add_argument("--predict-top-k", type=int, default=3)
    parser.add_argument("--top-k-paths", type=int, default=5)
    parser.add_argument("--top-k-shared", type=int, default=5)
    parser.add_argument("--top-k-analogies", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = build_explanation_context(snapshot_dir=args.snapshot_dir, dataset_dir=args.dataset_dir)
    result = explain_prediction(
        ctx,
        head=args.head,
        relation=args.relation,
        tail=args.tail,
        predict_top_k=args.predict_top_k,
        top_k_paths=args.top_k_paths,
        top_k_shared=args.top_k_shared,
        top_k_analogies=args.top_k_analogies,
    )
    result["snapshot_dir"] = str(args.snapshot_dir.resolve())
    result["dataset_dir"] = str(args.dataset_dir.resolve())

    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Query:", result["query"])
    print("Prediction:", result["prediction"])
    print("Summary:", result["summary"])
    print("Top supporting paths:")
    for row in result["supporting_paths"]:
        print(
            f" {row['rank']:>2}. {row['path_sentence']} | "
            f"score={row['explanation_score']:.4f} "
            f"rel={row['relation_relevance']:.4f} "
            f"freq={row['path_frequency']}"
        )
    print("Top shared neighbors:")
    for row in result["shared_neighbors"]:
        print(f" {row['rank']:>2}. {row['neighbor']} | score={row['shared_neighbor_score']:.4f}")
    print("Top analogical support:")
    for row in result["similar_entity_support"]:
        print(
            f" {row['rank']:>2}. {row['analog_head']} -- {row['analog_relation']} -- {row['analog_tail']} | "
            f"score={row['support_score']:.4f}"
        )


if __name__ == "__main__":
    main()
