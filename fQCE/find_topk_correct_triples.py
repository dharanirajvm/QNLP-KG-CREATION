import argparse
import csv
import json
import random
from pathlib import Path

import torch

from training_fb15k237 import QuantumKGE, setup_quantum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find triples where trained fQCE ranking is correct (e.g., Hits@K)."
    )
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-triples", type=int, default=300)
    parser.add_argument("--eval-candidates", type=int, default=1024)
    parser.add_argument(
        "--require-both",
        action="store_true",
        help="If set, keep only triples where both tail rank<=K and head rank<=K.",
    )
    parser.add_argument("--backend", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-limit", type=int, default=50)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_kg_line(line: str) -> tuple[str, str, str]:
    parts = line.strip().split("\t") if "\t" in line else line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid triple line: {line}")
    return parts[0], parts[1], parts[2]


def load_split_ids(
    dataset_dir: Path,
    split: str,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> list[tuple[int, int, int]]:
    path = dataset_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    triples: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, r, t = parse_kg_line(line)
        if h in entity_to_id and r in relation_to_id and t in entity_to_id:
            triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    return triples


def build_filter_maps(
    dataset_dir: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    all_true: set[tuple[int, int, int]] = set()
    for split in ("train", "valid", "test"):
        for triple in load_split_ids(dataset_dir, split, entity_to_id, relation_to_id):
            all_true.add(triple)
    tails: dict[tuple[int, int], set[int]] = {}
    heads: dict[tuple[int, int], set[int]] = {}
    for h, r, t in all_true:
        tails.setdefault((h, r), set()).add(t)
        heads.setdefault((r, t), set()).add(h)
    return tails, heads


def relation_to_phrase(rel: str) -> str:
    rel = rel.strip().strip("/")
    if not rel:
        return "related_to"
    return " ".join([x for x in rel.split("/") if x]).replace("_", " ")


def pretty(raw: str, labels: dict[str, str]) -> str:
    return labels.get(raw, labels.get(raw.lower(), raw))


def main() -> None:
    args = parse_args()
    rnd = random.Random(args.seed)

    snapshot_dir = args.snapshot_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()

    entity_to_id = load_json(snapshot_dir / "entity_to_id.json")
    relation_to_id = load_json(snapshot_dir / "relation_to_id.json")
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    labels = {}
    for p in (snapshot_dir / "labels_human.json", dataset_dir / "labels_human.json"):
        if p.exists():
            labels = load_json(p)
            break

    cfg = load_json(snapshot_dir / "config.json")
    cfg_args = cfg.get("args", {})
    num_qubits = int(cfg_args.get("num_qubits", 6))
    num_entities = int(cfg.get("num_entities", len(entity_to_id)))
    num_relations = int(cfg.get("num_relations", len(relation_to_id)))
    backend = args.backend or str(cfg_args.get("q_backend", "default.qubit"))

    setup_quantum(num_qubits, backend)
    model = QuantumKGE(num_entities=num_entities, num_relations=num_relations, num_qubits=num_qubits)
    state = torch.load(snapshot_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tails_filter, heads_filter = build_filter_maps(dataset_dir, entity_to_id, relation_to_id)
    triples = load_split_ids(dataset_dir, args.split, entity_to_id, relation_to_id)

    if args.max_triples > 0 and len(triples) > args.max_triples:
        triples = rnd.sample(triples, args.max_triples)

    print(
        f"Loaded split={args.split} triples={len(triples)} | top_k={args.top_k} "
        f"eval_candidates={args.eval_candidates} backend={backend}"
    )

    right_rows: list[dict] = []

    for h, r, t in triples:
        tail_cands = {t}
        head_cands = {h}
        while len(tail_cands) < min(args.eval_candidates, num_entities):
            tail_cands.add(rnd.randint(0, num_entities - 1))
        while len(head_cands) < min(args.eval_candidates, num_entities):
            head_cands.add(rnd.randint(0, num_entities - 1))

        with torch.no_grad():
            sp = model.relation_subject_state(h, r)
            tail_scores = {
                c: float(torch.real(torch.vdot(model.entity_state(c), sp)).item()) for c in tail_cands
            }

            tail_state = model.entity_state(t)
            head_scores = {}
            for c in head_cands:
                sp_c = model.relation_subject_state(c, r)
                head_scores[c] = float(torch.real(torch.vdot(tail_state, sp_c)).item())

        tail_true = tail_scores[t]
        tail_filtered = tails_filter.get((h, r), set())
        tail_rank = 1 + sum(
            1
            for c, s in tail_scores.items()
            if c != t and c not in tail_filtered and s > tail_true
        )

        head_true = head_scores[h]
        head_filtered = heads_filter.get((r, t), set())
        head_rank = 1 + sum(
            1
            for c, s in head_scores.items()
            if c != h and c not in head_filtered and s > head_true
        )

        tail_ok = tail_rank <= args.top_k
        head_ok = head_rank <= args.top_k
        keep = tail_ok and head_ok if args.require_both else (tail_ok or head_ok)
        if keep:
            h_raw = id_to_entity[h]
            r_raw = id_to_relation[r]
            t_raw = id_to_entity[t]
            row = {
                "head": pretty(h_raw, labels),
                "relation": pretty(r_raw, labels),
                "tail": pretty(t_raw, labels),
                "tail_rank": tail_rank,
                "head_rank": head_rank,
                "tail_hit": tail_ok,
                "head_hit": head_ok,
                "triple_sentence": f"{pretty(h_raw, labels)} -- "
                f"{pretty(r_raw, labels) if r_raw in labels else relation_to_phrase(r_raw)} -- "
                f"{pretty(t_raw, labels)}",
            }
            right_rows.append(row)

    out_csv = snapshot_dir / f"rank_correct_{args.split}_top{args.top_k}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "head",
                "relation",
                "tail",
                "tail_rank",
                "head_rank",
                "tail_hit",
                "head_hit",
                "triple_sentence",
            ],
        )
        w.writeheader()
        w.writerows(right_rows)

    total = len(triples)
    count = len(right_rows)
    print(f"Correct triples ({'both' if args.require_both else 'head or tail'} rank<=K): {count}/{total} ({count/max(1,total):.4f})")
    print(f"Saved: {out_csv}")
    print("\nExamples:")
    for i, row in enumerate(right_rows[: args.print_limit], start=1):
        print(
            f"{i}. {row['triple_sentence']} | tail_rank={row['tail_rank']} head_rank={row['head_rank']}"
        )


if __name__ == "__main__":
    main()

