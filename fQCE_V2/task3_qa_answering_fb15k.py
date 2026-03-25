#!/usr/bin/env python3
"""Task 3 for FB15k-237: QA over the strong FB15k snapshot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODULE1_DIR = THIS_DIR.parent / "fQCE" / "module1_kge"
if str(MODULE1_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE1_DIR))

from downstream_utils import detect_entities_in_query, detect_relation_from_query, load_context  # type: ignore


DEFAULT_SNAPSHOT_DIR = THIS_DIR.parent / "fQCE" / "inference_snapshots" / "quantum_fb15k237_20260308_174529_updated_20260310_193344"
DEFAULT_DATASET_DIR = THIS_DIR.parent / "fQCE" / "datasets" / "fb15k237"

QUESTION_PATTERNS = [
    (re.compile(r"where\s+was\s+(.+?)\s+born\??$", re.I), "/people/person/place_of_birth", "tail"),
    (re.compile(r"what\s+is\s+(.+?)'s\s+profession\??$", re.I), "/people/person/profession", "tail"),
    (re.compile(r"what\s+is\s+the\s+profession\s+of\s+(.+?)\??$", re.I), "/people/person/profession", "tail"),
    (re.compile(r"what\s+is\s+(.+?)'s\s+nationality\??$", re.I), "/people/person/nationality", "tail"),
    (re.compile(r"which\s+country\s+is\s+(.+?)\s+from\??$", re.I), "/people/person/nationality", "tail"),
    (re.compile(r"where\s+did\s+(.+?)\s+study\??$", re.I), "/education/educational_degree/people_with_this_degree./education/education/institution", "tail"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question answering over FB15k-237 using exact graph lookup + KGE fallback.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def parse_question(question: str) -> dict:
    for pattern, relation_raw, direction in QUESTION_PATTERNS:
        match = pattern.match(question.strip())
        if match:
            return {"entity_text": match.group(1).strip(), "relation_raw": relation_raw, "direction": direction}
    return {"entity_text": None, "relation_raw": None, "direction": None}


def answer_question_fb15k(ctx, question: str, top_k: int) -> dict:
    parsed = parse_question(question)
    if parsed["entity_text"] is None or parsed["relation_raw"] is None:
        entity_hits = detect_entities_in_query(question, ctx, limit=1)
        relation_raw = detect_relation_from_query(question, ctx.relation_to_id)
        if not entity_hits or relation_raw is None:
            raise ValueError("Could not parse question into a FB15k KG query.")
        parsed = {
            "entity_text": ctx.display(ctx.id_to_entity[entity_hits[0]]),
            "relation_raw": relation_raw,
            "direction": "tail",
        }

    entity_id = ctx.resolve_entity(parsed["entity_text"])
    relation_id = ctx.resolve_relation(parsed["relation_raw"])
    direction = parsed["direction"]

    if direction == "head":
        known_answers = [
            {"answer": ctx.display(ctx.id_to_entity[h]), "sentence": ctx.sentence_for_ids(h, relation_id, entity_id), "source": "known_kg"}
            for h, _, _ in ctx.known_answers(relation_id=relation_id, tail_id=entity_id)
        ]
        ranked = ctx.rank_heads(
            relation_id,
            entity_id,
            top_k=ctx.num_entities,
            exclude_known=False,
        )
    else:
        known_answers = [
            {"answer": ctx.display(ctx.id_to_entity[t]), "sentence": ctx.sentence_for_ids(entity_id, relation_id, t), "source": "known_kg"}
            for _, _, t in ctx.known_answers(head_id=entity_id, relation_id=relation_id)
        ]
        ranked = ctx.rank_tails(
            entity_id,
            relation_id,
            top_k=ctx.num_entities,
            exclude_known=False,
        )

    predicted_answers = []
    for row in ranked:
        predicted_answers.append(
            {
                "answer": row["head"] if direction == "head" else row["tail"],
                "sentence": row["sentence"],
                "score": row["score"],
                "source": "kge_completion",
            }
        )

    merged = []
    seen = set()
    for item in known_answers + predicted_answers:
        key = item["answer"].lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    return {
        "task": "qa_answering_fb15k237",
        "question": question,
        "parsed_query": {
            "head": ctx.display(ctx.id_to_entity[entity_id]) if direction != "head" else "?",
            "relation": ctx.display(ctx.id_to_relation[relation_id]),
            "tail": ctx.display(ctx.id_to_entity[entity_id]) if direction == "head" else "?",
        },
        "answers": merged[:top_k],
    }


def main() -> None:
    args = parse_args()
    ctx = load_context(snapshot_dir=args.snapshot_dir, dataset_dir=args.dataset_dir)
    result = answer_question_fb15k(ctx, args.question, args.top_k)
    result["snapshot_dir"] = str(args.snapshot_dir.resolve())
    result["dataset_dir"] = str(args.dataset_dir.resolve())

    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Question:", result["question"])
    parsed = result["parsed_query"]
    print(f"KG query: ({parsed['head']}, {parsed['relation']}, {parsed['tail']})")
    print("Answers:")
    for idx, answer in enumerate(result["answers"], start=1):
        score = answer.get("score")
        score_text = f" | score={score:.6f}" if score is not None else ""
        print(f" {idx:>2}. {answer['answer']} | source={answer['source']}{score_text}")


if __name__ == "__main__":
    main()
