#!/usr/bin/env python3
"""Task 2: semantic similarity over KGE entity embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from downstream_utils import load_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic similarity over module1 KGE entities.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--anchor", type=str, required=True, help="Anchor entity.")
    parser.add_argument("--target", type=str, default="", help="Optional second entity for pairwise cosine similarity.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def run_semantic_similarity(ctx, anchor: str, target: str, top_k: int, include_self: bool) -> dict:
    anchor_id = ctx.resolve_entity(anchor)
    result = {
        "task": "semantic_similarity",
        "anchor_raw": ctx.id_to_entity[anchor_id],
        "anchor": ctx.display(ctx.id_to_entity[anchor_id]),
    }

    if target.strip():
        target_id = ctx.resolve_entity(target)
        result["target_raw"] = ctx.id_to_entity[target_id]
        result["target"] = ctx.display(ctx.id_to_entity[target_id])
        result["cosine_similarity"] = ctx.similarity(anchor_id, target_id)

    result["neighbors"] = ctx.rank_similar_entities(
        anchor_id,
        top_k=top_k,
        exclude_self=not include_self,
    )
    return result


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
