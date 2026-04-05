#!/usr/bin/env python3
"""Export the full FB15k-237 dataset as a knowledge graph."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import networkx as nx

from explanation_utils import DEFAULT_DATASET_DIR, parse_kg_line


THIS_DIR = Path(__file__).resolve().parent
REVIEW3_ROOT = THIS_DIR.parent
DEFAULT_OUTPUT_DIR = REVIEW3_ROOT / "outputs" / "fb15k_full_kg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the full FB15k-237 dataset as a KG.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Dataset splits to include.",
    )
    parser.add_argument("--write-graphml", action="store_true", help="Write GraphML export.")
    parser.add_argument("--write-gexf", action="store_true", help="Write GEXF export.")
    parser.add_argument("--write-json-summary", action="store_true", help="Write JSON summary.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_label_file(path: Path) -> dict[str, str]:
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            data[parts[0]] = parts[1]
    return data


def build_full_graph(dataset_dir: Path, splits: list[str]):
    entity_text = load_label_file(dataset_dir / "entity2text.txt")
    relation_text = load_label_file(dataset_dir / "relation2text.txt")
    labels_human = {}
    labels_json = dataset_dir / "labels_human.json"
    if labels_json.exists():
        labels_human = json.loads(labels_json.read_text(encoding="utf-8"))

    graph = nx.MultiDiGraph()
    edge_rows = []
    relation_counts = Counter()
    entity_degree = Counter()

    for split in splits:
        split_path = dataset_dir / f"{split}.txt"
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            head_raw, relation_raw, tail_raw = parse_kg_line(line)
            head_text = labels_human.get(head_raw, entity_text.get(head_raw, head_raw))
            tail_text = labels_human.get(tail_raw, entity_text.get(tail_raw, tail_raw))
            relation_text_value = labels_human.get(relation_raw, relation_text.get(relation_raw, relation_raw))

            if head_raw not in graph:
                graph.add_node(head_raw, label=head_text, raw_id=head_raw)
            if tail_raw not in graph:
                graph.add_node(tail_raw, label=tail_text, raw_id=tail_raw)

            graph.add_edge(
                head_raw,
                tail_raw,
                relation_raw=relation_raw,
                relation=relation_text_value,
                split=split,
            )

            edge_rows.append(
                {
                    "head_raw": head_raw,
                    "head": head_text,
                    "relation_raw": relation_raw,
                    "relation": relation_text_value,
                    "tail_raw": tail_raw,
                    "tail": tail_text,
                    "split": split,
                }
            )
            relation_counts[relation_raw] += 1
            entity_degree[head_raw] += 1
            entity_degree[tail_raw] += 1

    node_rows = []
    for raw_id, attrs in graph.nodes(data=True):
        node_rows.append(
            {
                "entity_raw": raw_id,
                "entity": attrs.get("label", raw_id),
                "degree": int(entity_degree.get(raw_id, 0)),
            }
        )

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "splits": splits,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "top_relations": [
            {
                "relation_raw": rel_raw,
                "relation": labels_human.get(rel_raw, relation_text.get(rel_raw, rel_raw)),
                "count": count,
            }
            for rel_raw, count in relation_counts.most_common(25)
        ],
        "top_entities_by_degree": [
            {
                "entity_raw": ent_raw,
                "entity": labels_human.get(ent_raw, entity_text.get(ent_raw, ent_raw)),
                "degree": count,
            }
            for ent_raw, count in entity_degree.most_common(25)
        ],
    }
    return graph, node_rows, edge_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    graph, node_rows, edge_rows, summary = build_full_graph(args.dataset_dir, args.splits)

    write_csv(output_dir / "nodes.csv", node_rows)
    write_csv(output_dir / "edges.csv", edge_rows)

    (output_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.write_graphml:
        nx.write_graphml(graph, output_dir / "fb15k_full.graphml")

    if args.write_gexf:
        nx.write_gexf(graph, output_dir / "fb15k_full.gexf")

    print("Saved full KG export to:", output_dir)
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    if args.write_graphml:
        print("GraphML:", output_dir / "fb15k_full.graphml")
    if args.write_gexf:
        print("GEXF:", output_dir / "fb15k_full.gexf")


if __name__ == "__main__":
    main()
