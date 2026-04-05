#!/usr/bin/env python3
"""Lightweight FB15k-237 dataset explorer for choosing explainability examples."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

try:
    import networkx as nx
    from pyvis.network import Network
except Exception:  # noqa: BLE001
    nx = None
    Network = None

from explanation_utils import (
    DEFAULT_DATASET_DIR,
    DEFAULT_SNAPSHOT_DIR,
    build_explanation_context,
    extract_local_subgraph,
    resolve_entity_input,
    resolve_relation_input,
)


THIS_DIR = Path(__file__).resolve().parent
REVIEW3_ROOT = THIS_DIR.parent
DEFAULT_OUTPUT_DIR = REVIEW3_ROOT / "outputs" / "fb15k_explorer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FB15k-237 neighborhoods, triples, and relation slices.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--mode", choices=["overview", "entity", "triple", "relation"], default="entity")
    parser.add_argument("--entity", type=str, default="", help="Entity ID or text.")
    parser.add_argument("--head", type=str, default="", help="Head entity ID or text.")
    parser.add_argument("--relation", type=str, default="", help="Relation ID or text.")
    parser.add_argument("--tail", type=str, default="", help="Tail entity ID or text.")
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-branch", type=int, default=12)
    parser.add_argument("--max-edges", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "fb15k_view"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_graph_html(nodes: list[dict], edges: list[dict], output_html: Path, title: str) -> None:
    if Network is None or nx is None:
        items = [
            "<html><body>",
            f"<h2>{html.escape(title)}</h2>",
            "<h3>Nodes</h3><ul>",
        ]
        for node in nodes:
            items.append(f"<li>{html.escape(node['entity'])} (degree={node['degree']})</li>")
        items.append("</ul><h3>Edges</h3><ul>")
        for edge in edges:
            items.append(
                "<li>"
                f"{html.escape(edge['head'])} -- {html.escape(edge['relation'])} -- {html.escape(edge['tail'])}"
                "</li>"
            )
        items.append("</ul></body></html>")
        output_html.write_text("\n".join(items), encoding="utf-8")
        return

    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(
            node["entity_id"],
            label=node["entity"],
            title=f"{node['entity']}<br>degree={node['degree']}",
        )
    for edge in edges:
        graph.add_edge(edge["head_id"], edge["tail_id"], label=edge["relation"], title=edge["relation"])

    net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111111")
    net.from_nx(graph)
    for node in net.nodes:
        node["shape"] = "dot"
        node["size"] = 18
    for edge in net.edges:
        edge["arrows"] = "to"
        edge["color"] = "#5b7c99"
    net.write_html(str(output_html), notebook=False)


def relation_examples(ctx, relation_id: int, top_k: int) -> dict:
    triples = [(h, r, t) for h, r, t in ctx.train_triples if r == relation_id]
    head_counts = Counter(h for h, _, _ in triples)
    tail_counts = Counter(t for _, _, t in triples)
    sample_rows = [
        {
            "head": ctx.kge.display(ctx.kge.id_to_entity[h]),
            "relation": ctx.kge.display(ctx.kge.id_to_relation[r]),
            "tail": ctx.kge.display(ctx.kge.id_to_entity[t]),
        }
        for h, r, t in triples[:top_k]
    ]
    return {
        "relation": ctx.kge.display(ctx.kge.id_to_relation[relation_id]),
        "raw_relation": ctx.kge.id_to_relation[relation_id],
        "train_count": len(triples),
        "top_heads": [
            {"entity": ctx.kge.display(ctx.kge.id_to_entity[entity_id]), "count": count}
            for entity_id, count in head_counts.most_common(top_k)
        ],
        "top_tails": [
            {"entity": ctx.kge.display(ctx.kge.id_to_entity[entity_id]), "count": count}
            for entity_id, count in tail_counts.most_common(top_k)
        ],
        "sample_triples": sample_rows,
    }


def dataset_overview(ctx, top_k: int) -> dict:
    relation_counts = Counter(r for _, r, _ in ctx.train_triples)
    entity_degree = Counter()
    for h, _, t in ctx.train_triples:
        entity_degree[h] += 1
        entity_degree[t] += 1
    return {
        "num_train_triples": len(ctx.train_triples),
        "num_entities": ctx.kge.num_entities,
        "num_relations": ctx.kge.num_relations,
        "top_relations": [
            {
                "relation": ctx.kge.display(ctx.kge.id_to_relation[rel_id]),
                "raw_relation": ctx.kge.id_to_relation[rel_id],
                "count": count,
            }
            for rel_id, count in relation_counts.most_common(top_k)
        ],
        "top_entities_by_degree": [
            {
                "entity": ctx.kge.display(ctx.kge.id_to_entity[entity_id]),
                "raw_entity": ctx.kge.id_to_entity[entity_id],
                "degree": count,
            }
            for entity_id, count in entity_degree.most_common(top_k)
        ],
    }


def main() -> None:
    args = parse_args()
    ctx = build_explanation_context(snapshot_dir=args.snapshot_dir, dataset_dir=args.dataset_dir)
    out_root = ensure_dir(args.output_dir)

    if args.mode == "overview":
        overview = dataset_overview(ctx, args.top_k)
        out_dir = ensure_dir(out_root / "overview")
        write_json(out_dir / "dataset_overview.json", overview)
        print("Saved dataset overview to", out_dir / "dataset_overview.json")
        return

    if args.mode == "relation":
        if not args.relation.strip():
            raise SystemExit("--mode relation requires --relation.")
        relation_id = resolve_relation_input(ctx, args.relation)
        payload = relation_examples(ctx, relation_id, args.top_k)
        out_dir = ensure_dir(out_root / f"relation_{safe_name(payload['relation'])}")
        write_json(out_dir / "relation_summary.json", payload)
        write_csv(out_dir / "sample_triples.csv", payload["sample_triples"])
        print("Saved relation summary to", out_dir / "relation_summary.json")
        return

    if args.mode == "entity":
        if not args.entity.strip():
            raise SystemExit("--mode entity requires --entity.")
        entity_id = resolve_entity_input(ctx, args.entity)
        entity_name = ctx.kge.display(ctx.kge.id_to_entity[entity_id])
        subgraph = extract_local_subgraph(
            ctx,
            entity_id,
            entity_id,
            max_hops=args.max_hops,
            max_branch=args.max_branch,
            max_edges=args.max_edges,
        )
        out_dir = ensure_dir(out_root / f"entity_{safe_name(entity_name)}")
        write_json(out_dir / "subgraph.json", subgraph)
        write_csv(out_dir / "nodes.csv", subgraph["nodes"])
        write_csv(out_dir / "edges.csv", subgraph["edges"])
        render_graph_html(subgraph["nodes"], subgraph["edges"], out_dir / "subgraph.html", f"Entity neighborhood: {entity_name}")
        print("Saved entity neighborhood to", out_dir)
        return

    if not args.head.strip() or not args.relation.strip():
        raise SystemExit("--mode triple requires --head and --relation.")

    head_id = resolve_entity_input(ctx, args.head)
    relation_id = resolve_relation_input(ctx, args.relation)
    if args.tail.strip():
        tail_id = resolve_entity_input(ctx, args.tail)
    else:
        ranked = ctx.kge.rank_tails(head_id, relation_id, top_k=1, exclude_known=False)
        if not ranked:
            raise SystemExit("No tail predictions found for the given query.")
        tail_id = int(ranked[0]["tail_id"])

    head_name = ctx.kge.display(ctx.kge.id_to_entity[head_id])
    relation_name = ctx.kge.display(ctx.kge.id_to_relation[relation_id])
    tail_name = ctx.kge.display(ctx.kge.id_to_entity[tail_id])
    subgraph = extract_local_subgraph(
        ctx,
        head_id,
        tail_id,
        max_hops=args.max_hops,
        max_branch=args.max_branch,
        max_edges=args.max_edges,
    )
    out_dir = ensure_dir(out_root / f"triple_{safe_name(head_name)}_{safe_name(relation_name)}_{safe_name(tail_name)}")
    payload = {
        "query": {"head": head_name, "relation": relation_name, "tail": tail_name},
        "subgraph": subgraph,
    }
    write_json(out_dir / "triple_subgraph.json", payload)
    write_csv(out_dir / "nodes.csv", subgraph["nodes"])
    write_csv(out_dir / "edges.csv", subgraph["edges"])
    render_graph_html(
        subgraph["nodes"],
        subgraph["edges"],
        out_dir / "subgraph.html",
        f"Triple neighborhood: {head_name} -- {relation_name} -- {tail_name}",
    )
    print("Saved triple neighborhood to", out_dir)


if __name__ == "__main__":
    main()
