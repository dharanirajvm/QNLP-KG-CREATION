#!/usr/bin/env python3
"""Task 3: Natural-language QA over module1 KG with KGE fallback."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from downstream_utils import detect_entities_in_query, detect_relation_from_query, load_context, relation_aliases


QUESTION_PATTERNS = [
    (re.compile(r"where\s+did\s+(.+?)\s+study\??$", re.I), "studies_at", "tail"),
    (re.compile(r"where\s+does\s+(.+?)\s+study\??$", re.I), "studies_at", "tail"),
    (re.compile(r"where\s+does\s+(.+?)\s+work\??$", re.I), "works_at", "tail"),
    (re.compile(r"where\s+did\s+(.+?)\s+work\??$", re.I), "works_at", "tail"),
    (re.compile(r"where\s+does\s+(.+?)\s+live\??$", re.I), "lives_in", "tail"),
    (re.compile(r"where\s+was\s+(.+?)\s+born\??$", re.I), "born_in", "tail"),
    (re.compile(r"what\s+does\s+(.+?)\s+use\??$", re.I), "uses", "tail"),
    (re.compile(r"what\s+does\s+(.+?)\s+own\??$", re.I), "owns", "tail"),
    (re.compile(r"who\s+works\s+at\s+(.+?)\??$", re.I), "works_at", "head"),
    (re.compile(r"who\s+studies\s+at\s+(.+?)\??$", re.I), "studies_at", "head"),
    (re.compile(r"who\s+lives\s+in\s+(.+?)\??$", re.I), "lives_in", "head"),
    (re.compile(r"what\s+is\s+located\s+in\s+(.+?)\??$", re.I), "located_in", "head"),
    (re.compile(r"who\s+is\s+married\s+to\s+(.+?)\??$", re.I), "married_to", "head"),
    (re.compile(r"who\s+is\s+the\s+parent\s+of\s+(.+?)\??$", re.I), "parent_of", "head"),
    (re.compile(r"who\s+are\s+the\s+parents\s+of\s+(.+?)\??$", re.I), "parent_of", "head"),
    (re.compile(r"who\s+are\s+the\s+children\s+of\s+(.+?)\??$", re.I), "parent_of", "tail_from_head"),
    (re.compile(r"what\s+is\s+part\s+of\s+(.+?)\??$", re.I), "part_of", "head"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA answering over module1 KG using exact graph answers + KGE completion.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def parse_question(question: str) -> dict:
    q = question.strip()
    for pattern, relation_raw, direction in QUESTION_PATTERNS:
        match = pattern.match(q)
        if match:
            return {"relation_raw": relation_raw, "entity_text": match.group(1).strip(), "direction": direction}
    return {"relation_raw": None, "entity_text": None, "direction": None}


def fallback_parse(ctx, question: str) -> dict:
    relation_raw = detect_relation_from_query(question, ctx.relation_to_id)
    entity_hits = detect_entities_in_query(question, ctx, limit=1)
    direction = "tail"
    q = question.lower()
    if any(token in q for token in ["who ", "which ", "what is located in", "what is part of"]):
        direction = "head"
    if any(token in q for token in ["children of"]):
        direction = "tail_from_head"
    entity_text = ctx.display(ctx.id_to_entity[entity_hits[0]]) if entity_hits else None
    return {"relation_raw": relation_raw, "entity_text": entity_text, "direction": direction}


def answer_question(ctx, question: str, top_k: int) -> dict:
    parsed = parse_question(question)
    if parsed["relation_raw"] is None or parsed["entity_text"] is None:
        parsed = fallback_parse(ctx, question)
    if parsed["relation_raw"] is None or parsed["entity_text"] is None:
        raise ValueError("Could not parse question into a KG query.")

    relation_id = ctx.resolve_relation(parsed["relation_raw"])
    entity_id = ctx.resolve_entity(parsed["entity_text"])
    direction = parsed["direction"]

    known_answers = []
    predicted = []
    if direction == "head":
        known_answers = [
            {
                "answer": ctx.display(ctx.id_to_entity[h]),
                "sentence": ctx.sentence_for_ids(h, relation_id, entity_id),
                "source": "known_kg",
            }
            for h, _, _ in ctx.known_answers(relation_id=relation_id, tail_id=entity_id)
        ]
        predicted = ctx.rank_heads(relation_id, entity_id, top_k=top_k, exclude_known=False)
    elif direction == "tail_from_head":
        known_answers = [
            {
                "answer": ctx.display(ctx.id_to_entity[t]),
                "sentence": ctx.sentence_for_ids(entity_id, relation_id, t),
                "source": "known_kg",
            }
            for _, _, t in ctx.known_answers(head_id=entity_id, relation_id=relation_id)
        ]
        predicted = ctx.rank_tails(entity_id, relation_id, top_k=top_k, exclude_known=False)
    else:
        known_answers = [
            {
                "answer": ctx.display(ctx.id_to_entity[t]),
                "sentence": ctx.sentence_for_ids(entity_id, relation_id, t),
                "source": "known_kg",
            }
            for _, _, t in ctx.known_answers(head_id=entity_id, relation_id=relation_id)
        ]
        predicted = ctx.rank_tails(entity_id, relation_id, top_k=top_k, exclude_known=False)

    formatted_predictions = []
    for row in predicted:
        if direction == "head":
            answer = row["head"]
        else:
            answer = row["tail"]
        formatted_predictions.append(
            {
                "answer": answer,
                "score": row["score"],
                "rank": row["rank"],
                "sentence": row["sentence"],
                "source": "kge_completion",
            }
        )

    dedup_seen = set()
    merged_answers = []
    for item in known_answers + formatted_predictions:
        key = item["answer"].lower()
        if key in dedup_seen:
            continue
        dedup_seen.add(key)
        merged_answers.append(item)

    relation_raw = ctx.id_to_relation[relation_id]
    query_template = {
        "head": "?" if direction == "head" else ctx.display(ctx.id_to_entity[entity_id]),
        "relation": ctx.display(relation_raw),
        "tail": ctx.display(ctx.id_to_entity[entity_id]) if direction == "head" else "?",
    }
    if direction == "tail_from_head":
        query_template["head"] = ctx.display(ctx.id_to_entity[entity_id])
        query_template["tail"] = "?"

    return {
        "task": "qa_answering",
        "question": question,
        "parsed_query": query_template,
        "relation_aliases_used": sorted(relation_aliases(relation_raw)),
        "answers": merged_answers[:top_k],
    }


def main() -> None:
    args = parse_args()
    ctx = load_context(snapshot_dir=args.snapshot_dir, dataset_dir=args.dataset_dir)
    result = answer_question(ctx, args.question, args.top_k)

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

