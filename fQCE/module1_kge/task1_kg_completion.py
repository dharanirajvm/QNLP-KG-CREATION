#!/usr/bin/env python3
"""Task 1: Knowledge graph completion with a trained module1 KGE model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from downstream_utils import load_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge graph completion for module1 KGE.")
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Trained run directory.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset dir with train/valid/test splits.")
    parser.add_argument(
        "--mode",
        choices=["tail", "head", "relation", "batch"],
        default="tail",
        help="tail: (h,r,?), head: (?,r,t), relation: (h,?,t), batch: queries from file with one ? slot.",
    )
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--relation", type=str, default="")
    parser.add_argument("--tail", type=str, default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-known", action="store_true", help="Do not filter known true triples.")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Batch file with tab-separated triples and exactly one ? placeholder.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON results.")
    return parser.parse_args()


def batch_queries(path: Path) -> list[tuple[str, str, str]]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid batch query line: {line}")
        queries.append((parts[0], parts[1], parts[2]))
    return queries


def run_single(ctx, mode: str, head: str, relation: str, tail: str, top_k: int, include_known: bool) -> dict:
    if mode == "tail":
        head_id = ctx.resolve_entity(head)
        relation_id = ctx.resolve_relation(relation)
        rows = ctx.rank_tails(head_id, relation_id, top_k=top_k, exclude_known=not include_known)
        known = [
            {
                "tail": ctx.display(ctx.id_to_entity[t]),
                "sentence": ctx.sentence_for_ids(head_id, relation_id, t),
            }
            for _, _, t in ctx.known_answers(head_id=head_id, relation_id=relation_id)
        ]
        return {
            "mode": mode,
            "query": {"head": ctx.display(ctx.id_to_entity[head_id]), "relation": ctx.display(ctx.id_to_relation[relation_id]), "tail": "?"},
            "known_answers": known,
            "predictions": rows,
        }

    if mode == "head":
        relation_id = ctx.resolve_relation(relation)
        tail_id = ctx.resolve_entity(tail)
        rows = ctx.rank_heads(relation_id, tail_id, top_k=top_k, exclude_known=not include_known)
        known = [
            {
                "head": ctx.display(ctx.id_to_entity[h]),
                "sentence": ctx.sentence_for_ids(h, relation_id, tail_id),
            }
            for h, _, _ in ctx.known_answers(relation_id=relation_id, tail_id=tail_id)
        ]
        return {
            "mode": mode,
            "query": {"head": "?", "relation": ctx.display(ctx.id_to_relation[relation_id]), "tail": ctx.display(ctx.id_to_entity[tail_id])},
            "known_answers": known,
            "predictions": rows,
        }

    head_id = ctx.resolve_entity(head)
    tail_id = ctx.resolve_entity(tail)
    rows = ctx.rank_relations(head_id, tail_id, top_k=top_k)
    known = [
        {
            "relation": ctx.display(ctx.id_to_relation[r]),
            "sentence": ctx.sentence_for_ids(head_id, r, tail_id),
        }
        for _, r, _ in ctx.known_answers(head_id=head_id, tail_id=tail_id)
    ]
    return {
        "mode": mode,
        "query": {"head": ctx.display(ctx.id_to_entity[head_id]), "relation": "?", "tail": ctx.display(ctx.id_to_entity[tail_id])},
        "known_answers": known,
        "predictions": rows,
    }


def print_single(result: dict) -> None:
    query = result["query"]
    print(f"Query: ({query['head']}, {query['relation']}, {query['tail']})")
    if result["known_answers"]:
        print("Known answers:")
        for row in result["known_answers"][:10]:
            print(" -", row["sentence"])
    print("Predictions:")
    for row in result["predictions"]:
        if result["mode"] == "tail":
            answer = row["tail"]
        elif result["mode"] == "head":
            answer = row["head"]
        else:
            answer = row["relation"]
        print(f" {row['rank']:>2}. {answer} | score={row['score']:.6f}")


def main() -> None:
    args = parse_args()
    ctx = load_context(snapshot_dir=args.snapshot_dir, dataset_dir=args.dataset_dir)

    if args.mode == "batch":
        if args.queries_file is None:
            raise SystemExit("--mode batch requires --queries-file.")
        results = []
        for head, relation, tail in batch_queries(args.queries_file):
            slots_missing = [head == "?", relation == "?", tail == "?"]
            if sum(slots_missing) != 1:
                raise ValueError(f"Batch query must have exactly one ?: {(head, relation, tail)}")
            if head == "?":
                mode = "head"
            elif relation == "?":
                mode = "relation"
            else:
                mode = "tail"
            results.append(
                run_single(
                    ctx=ctx,
                    mode=mode,
                    head=head,
                    relation=relation,
                    tail=tail,
                    top_k=args.top_k,
                    include_known=args.include_known,
                )
            )
        payload = {"task": "knowledge_graph_completion", "results": results}
        if args.output_json:
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for item in results:
            print_single(item)
            print()
        return

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

