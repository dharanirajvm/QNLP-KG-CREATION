from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
MODULE1_DIR = PROJECT_ROOT / "fQCE" / "module1_kge"
if str(MODULE1_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE1_DIR))

from downstream_utils import KGEContext, humanize, load_context, norm_text, parse_kg_line  # type: ignore


DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "fQCE" / "inference_snapshots" / "quantum_fb15k237_20260308_174529_updated_20260310_193344"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "fQCE" / "datasets" / "fb15k237"


@dataclass
class ExplanationContext:
    kge: KGEContext
    train_triples: list[tuple[int, int, int]]
    outgoing: dict[int, list[tuple[int, int]]]
    incoming: dict[int, list[tuple[int, int]]]
    outgoing_by_relation: dict[int, dict[int, list[int]]]
    undirected_neighbors: dict[int, set[int]]
    pair_to_relations: dict[tuple[int, int], set[int]]
    pattern_instance_counts: Counter[tuple[int, int]]
    pattern_relation_support: Counter[tuple[tuple[int, int], int]]
    entity_text_to_raw: dict[str, set[str]]
    relation_text_to_raw: dict[str, set[str]]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_triples(dataset_dir: Path, entity_to_id: dict[str, int], relation_to_id: dict[str, int]) -> list[tuple[int, int, int]]:
    path = dataset_dir / "train.txt"
    triples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, r, t = parse_kg_line(line)
        if h in entity_to_id and r in relation_to_id and t in entity_to_id:
            triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    return triples


def load_label_file(path: Path) -> dict[str, str]:
    labels = {}
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            labels[parts[0]] = parts[1]
    return labels


def relation_aliases(raw_relation: str, display_text: str) -> set[str]:
    aliases = {
        norm_text(raw_relation),
        norm_text(display_text),
        norm_text(humanize(raw_relation)),
    }
    segments = [seg for seg in raw_relation.strip("/").split("/") if seg]
    cleaned_segments = [norm_text(seg.replace(".", " ").replace("_", " ")) for seg in segments]
    for width in (1, 2, 3):
        if len(cleaned_segments) >= width:
            aliases.add(norm_text(" ".join(cleaned_segments[-width:])))
    return {alias for alias in aliases if alias}


def build_text_lookup(
    raw_to_text: dict[str, str],
    *,
    is_relation: bool,
) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for raw, text in raw_to_text.items():
        aliases = relation_aliases(raw, text) if is_relation else {
            norm_text(raw),
            norm_text(text),
        }
        for alias in aliases:
            if alias:
                lookup[alias].add(raw)
    return dict(lookup)


def resolve_with_lookup(text: str, lookup: dict[str, set[str]], raw_to_id: dict[str, int], label_scope: str) -> int:
    s = norm_text(text)
    if not s:
        raise ValueError(f"Missing {label_scope}")

    if s in lookup:
        raws = sorted(lookup[s])
        if len(raws) == 1:
            return raw_to_id[raws[0]]
        raise KeyError(f"Ambiguous {label_scope}: {text}")

    substring_hits = []
    for alias, raws in lookup.items():
        if s in alias or alias in s:
            substring_hits.extend(sorted(raws))
    substring_hits = sorted(set(substring_hits))
    if len(substring_hits) == 1:
        return raw_to_id[substring_hits[0]]
    if substring_hits:
        raise KeyError(f"Ambiguous {label_scope}: {text}")
    raise KeyError(f"Unknown {label_scope}: {text}")


def resolve_entity_input(ctx: ExplanationContext, text: str) -> int:
    try:
        return ctx.kge.resolve_entity(text)
    except Exception:
        return resolve_with_lookup(text, ctx.entity_text_to_raw, ctx.kge.entity_to_id, "entity")


def resolve_relation_input(ctx: ExplanationContext, text: str) -> int:
    try:
        return ctx.kge.resolve_relation(text)
    except Exception:
        return resolve_with_lookup(text, ctx.relation_text_to_raw, ctx.kge.relation_to_id, "relation")


def build_explanation_context(
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> ExplanationContext:
    kge = load_context(snapshot_dir=snapshot_dir, dataset_dir=dataset_dir)
    train_triples = load_train_triples(dataset_dir, kge.entity_to_id, kge.relation_to_id)
    entity_text = load_label_file(dataset_dir / "entity2text.txt")
    relation_text = load_label_file(dataset_dir / "relation2text.txt")
    raw_entity_text = {raw: entity_text.get(raw, kge.display(raw)) for raw in kge.entity_to_id}
    raw_relation_text = {raw: relation_text.get(raw, kge.display(raw)) for raw in kge.relation_to_id}

    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    incoming: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing_by_relation: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    undirected_neighbors: dict[int, set[int]] = defaultdict(set)
    pair_to_relations: dict[tuple[int, int], set[int]] = defaultdict(set)

    for h, r, t in train_triples:
        outgoing[h].append((r, t))
        incoming[t].append((r, h))
        outgoing_by_relation[h][r].append(t)
        undirected_neighbors[h].add(t)
        undirected_neighbors[t].add(h)
        pair_to_relations[(h, t)].add(r)

    pattern_instance_counts: Counter[tuple[int, int]] = Counter()
    pattern_relation_support: Counter[tuple[tuple[int, int], int]] = Counter()
    for h, r1, mid in train_triples:
        for r2, t in outgoing.get(mid, []):
            pattern = (r1, r2)
            pattern_instance_counts[pattern] += 1
            for target_relation in pair_to_relations.get((h, t), set()):
                pattern_relation_support[(pattern, target_relation)] += 1

    return ExplanationContext(
        kge=kge,
        train_triples=train_triples,
        outgoing=dict(outgoing),
        incoming=dict(incoming),
        outgoing_by_relation={node: dict(rel_map) for node, rel_map in outgoing_by_relation.items()},
        undirected_neighbors=dict(undirected_neighbors),
        pair_to_relations=dict(pair_to_relations),
        pattern_instance_counts=pattern_instance_counts,
        pattern_relation_support=pattern_relation_support,
        entity_text_to_raw=build_text_lookup(raw_entity_text, is_relation=False),
        relation_text_to_raw=build_text_lookup(raw_relation_text, is_relation=True),
    )


def normalize_rows(rows: list[dict], key: str, out_key: str) -> None:
    if not rows:
        return
    vals = [float(row[key]) for row in rows]
    lo = min(vals)
    hi = max(vals)
    if hi - lo < 1e-9:
        for row in rows:
            row[out_key] = 1.0 if hi > 0 else 0.0
        return
    for row in rows:
        row[out_key] = (float(row[key]) - lo) / (hi - lo)


def edge_count(ctx: ExplanationContext, node_id: int) -> int:
    return len(ctx.outgoing.get(node_id, [])) + len(ctx.incoming.get(node_id, []))


def extract_local_subgraph(
    ctx: ExplanationContext,
    head_id: int,
    tail_id: int,
    *,
    max_hops: int = 2,
    max_branch: int = 20,
    max_edges: int = 100,
) -> dict:
    visited = {head_id, tail_id}
    frontier = {head_id, tail_id}

    for _ in range(max_hops):
        nxt = set()
        for node_id in frontier:
            neighbors = sorted(
                ctx.undirected_neighbors.get(node_id, set()),
                key=lambda nid: (edge_count(ctx, nid), nid),
            )[:max_branch]
            for nbr in neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    nxt.add(nbr)
        frontier = nxt
        if not frontier:
            break

    edges = []
    for h, r, t in ctx.train_triples:
        if h in visited and t in visited:
            edges.append(
                {
                    "head_id": h,
                    "head": ctx.kge.display(ctx.kge.id_to_entity[h]),
                    "relation_id": r,
                    "relation": ctx.kge.display(ctx.kge.id_to_relation[r]),
                    "tail_id": t,
                    "tail": ctx.kge.display(ctx.kge.id_to_entity[t]),
                }
            )
            if len(edges) >= max_edges:
                break

    nodes = [
        {
            "entity_id": entity_id,
            "entity_raw": ctx.kge.id_to_entity[entity_id],
            "entity": ctx.kge.display(ctx.kge.id_to_entity[entity_id]),
            "degree": edge_count(ctx, entity_id),
        }
        for entity_id in sorted(visited)
    ]
    return {"nodes": nodes, "edges": edges}


def enumerate_two_hop_paths(
    ctx: ExplanationContext,
    head_id: int,
    relation_id: int,
    tail_id: int,
    *,
    top_k: int = 10,
) -> list[dict]:
    rows = []
    for r1, mid_id in ctx.outgoing.get(head_id, []):
        for r2, end_id in ctx.outgoing.get(mid_id, []):
            if end_id != tail_id:
                continue

            pattern = (r1, r2)
            path_reliability = 1.0 / max(1, len(ctx.outgoing_by_relation.get(head_id, {}).get(r1, [])))
            path_reliability *= 1.0 / max(1, len(ctx.outgoing_by_relation.get(mid_id, {}).get(r2, [])))

            pattern_count = int(ctx.pattern_instance_counts.get(pattern, 0))
            relation_support = int(ctx.pattern_relation_support.get((pattern, relation_id), 0))
            relation_relevance = relation_support / pattern_count if pattern_count else 0.0
            embedding_support = 0.5 * (
                ctx.kge.similarity(head_id, mid_id) + ctx.kge.similarity(mid_id, tail_id)
            )

            rows.append(
                {
                    "intermediate_id": mid_id,
                    "intermediate": ctx.kge.display(ctx.kge.id_to_entity[mid_id]),
                    "path_pattern_raw": [ctx.kge.id_to_relation[r1], ctx.kge.id_to_relation[r2]],
                    "path_pattern": [
                        ctx.kge.display(ctx.kge.id_to_relation[r1]),
                        ctx.kge.display(ctx.kge.id_to_relation[r2]),
                    ],
                    "path_sentence": (
                        f"{ctx.kge.display(ctx.kge.id_to_entity[head_id])} -- "
                        f"{ctx.kge.display(ctx.kge.id_to_relation[r1])} -- "
                        f"{ctx.kge.display(ctx.kge.id_to_entity[mid_id])} -- "
                        f"{ctx.kge.display(ctx.kge.id_to_relation[r2])} -- "
                        f"{ctx.kge.display(ctx.kge.id_to_entity[tail_id])}"
                    ),
                    "path_reliability": path_reliability,
                    "relation_relevance": relation_relevance,
                    "path_frequency": pattern_count,
                    "embedding_support": embedding_support,
                }
            )

    if not rows:
        return rows

    normalize_rows(rows, "path_frequency", "path_frequency_norm")
    normalize_rows(rows, "embedding_support", "embedding_support_norm")
    for row in rows:
        row["explanation_score"] = (
            0.40 * float(row["path_reliability"])
            + 0.30 * float(row["relation_relevance"])
            + 0.15 * float(row["path_frequency_norm"])
            + 0.15 * float(row["embedding_support_norm"])
        )
    rows.sort(key=lambda row: row["explanation_score"], reverse=True)
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def collect_shared_neighbors(
    ctx: ExplanationContext,
    head_id: int,
    relation_id: int,
    tail_id: int,
    *,
    top_k: int = 10,
) -> list[dict]:
    shared = ctx.undirected_neighbors.get(head_id, set()) & ctx.undirected_neighbors.get(tail_id, set())
    rows = []
    target_rel_raw = ctx.kge.id_to_relation[relation_id]

    for nbr_id in shared:
        head_edges = [r for r, dst in ctx.outgoing.get(head_id, []) if dst == nbr_id]
        tail_edges = [r for r, dst in ctx.outgoing.get(tail_id, []) if dst == nbr_id]
        in_to_head = [r for r, src in ctx.incoming.get(head_id, []) if src == nbr_id]
        in_to_tail = [r for r, src in ctx.incoming.get(tail_id, []) if src == nbr_id]

        relation_match_strength = 0.0
        if relation_id in head_edges or relation_id in tail_edges or relation_id in in_to_head or relation_id in in_to_tail:
            relation_match_strength = 1.0
        elif any(ctx.kge.id_to_relation[r] == target_rel_raw for r in head_edges + tail_edges + in_to_head + in_to_tail):
            relation_match_strength = 0.5

        degree_penalty = 1.0 / math.log2(2.0 + edge_count(ctx, nbr_id))
        embedding_coherence = 0.5 * (
            ctx.kge.similarity(head_id, nbr_id) + ctx.kge.similarity(tail_id, nbr_id)
        )
        score = 0.55 * embedding_coherence + 0.25 * degree_penalty + 0.20 * relation_match_strength

        rows.append(
            {
                "neighbor_id": nbr_id,
                "neighbor": ctx.kge.display(ctx.kge.id_to_entity[nbr_id]),
                "head_to_neighbor_relations": [ctx.kge.display(ctx.kge.id_to_relation[r]) for r in head_edges],
                "neighbor_to_head_relations": [ctx.kge.display(ctx.kge.id_to_relation[r]) for r in in_to_head],
                "tail_to_neighbor_relations": [ctx.kge.display(ctx.kge.id_to_relation[r]) for r in tail_edges],
                "neighbor_to_tail_relations": [ctx.kge.display(ctx.kge.id_to_relation[r]) for r in in_to_tail],
                "degree_penalty": degree_penalty,
                "relation_match_strength": relation_match_strength,
                "embedding_coherence": embedding_coherence,
                "shared_neighbor_score": score,
            }
        )

    rows.sort(key=lambda row: row["shared_neighbor_score"], reverse=True)
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def collect_analogical_support(
    ctx: ExplanationContext,
    head_id: int,
    relation_id: int,
    tail_id: int,
    *,
    top_k: int = 10,
    similarity_pool: int = 15,
) -> list[dict]:
    rows = []
    seen = set()

    similar_heads = ctx.kge.rank_similar_entities(head_id, top_k=similarity_pool, exclude_self=True)
    for item in similar_heads:
        analog_head_id = int(item["neighbor_id"])
        head_similarity = float(item["cosine_similarity"])
        for rel_id, analog_tail_id in ctx.outgoing.get(analog_head_id, []):
            if rel_id != relation_id:
                continue
            tail_similarity = ctx.kge.similarity(tail_id, analog_tail_id)
            support_score = 0.65 * head_similarity + 0.35 * tail_similarity
            key = (analog_head_id, rel_id, analog_tail_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "support_type": "similar_head",
                    "analog_head": ctx.kge.display(ctx.kge.id_to_entity[analog_head_id]),
                    "analog_relation": ctx.kge.display(ctx.kge.id_to_relation[rel_id]),
                    "analog_tail": ctx.kge.display(ctx.kge.id_to_entity[analog_tail_id]),
                    "head_similarity": head_similarity,
                    "tail_similarity": tail_similarity,
                    "support_score": support_score,
                }
            )

    similar_tails = ctx.kge.rank_similar_entities(tail_id, top_k=similarity_pool, exclude_self=True)
    for item in similar_tails:
        analog_tail_id = int(item["neighbor_id"])
        tail_similarity = float(item["cosine_similarity"])
        for rel_id, analog_head_id in ctx.incoming.get(analog_tail_id, []):
            if rel_id != relation_id:
                continue
            head_similarity = ctx.kge.similarity(head_id, analog_head_id)
            support_score = 0.35 * head_similarity + 0.65 * tail_similarity
            key = (analog_head_id, rel_id, analog_tail_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "support_type": "similar_tail",
                    "analog_head": ctx.kge.display(ctx.kge.id_to_entity[analog_head_id]),
                    "analog_relation": ctx.kge.display(ctx.kge.id_to_relation[rel_id]),
                    "analog_tail": ctx.kge.display(ctx.kge.id_to_entity[analog_tail_id]),
                    "head_similarity": head_similarity,
                    "tail_similarity": tail_similarity,
                    "support_score": support_score,
                }
            )

    rows.sort(key=lambda row: row["support_score"], reverse=True)
    for rank, row in enumerate(rows[:top_k], start=1):
        row["rank"] = rank
    return rows[:top_k]


def build_summary(
    ctx: ExplanationContext,
    head_id: int,
    relation_id: int,
    tail_id: int,
    paths: list[dict],
    shared_neighbors: list[dict],
    analogies: list[dict],
) -> str:
    head = ctx.kge.display(ctx.kge.id_to_entity[head_id])
    relation = ctx.kge.display(ctx.kge.id_to_relation[relation_id])
    tail = ctx.kge.display(ctx.kge.id_to_entity[tail_id])

    if paths:
        best = paths[0]
        path_text = " then ".join(best["path_pattern"])
        return (
            f"The prediction {head} -- {relation} -- {tail} is mainly supported by the "
            f"two-hop path through {best['intermediate']} via {path_text}."
        )
    if shared_neighbors:
        best = shared_neighbors[0]
        return (
            f"The prediction {head} -- {relation} -- {tail} is supported by the shared "
            f"neighbor {best['neighbor']}, which connects structurally to both entities."
        )
    if analogies:
        best = analogies[0]
        return (
            f"The prediction {head} -- {relation} -- {tail} is supported analogically by "
            f"the known triple {best['analog_head']} -- {best['analog_relation']} -- {best['analog_tail']}."
        )
    return (
        f"No strong short-path explanation was found for {head} -- {relation} -- {tail}; "
        "the prediction is currently supported only by the latent KGE score."
    )


def explain_prediction(
    ctx: ExplanationContext,
    *,
    head: str,
    relation: str,
    tail: str = "",
    predict_top_k: int = 3,
    top_k_paths: int = 5,
    top_k_shared: int = 5,
    top_k_analogies: int = 5,
) -> dict:
    head_id = resolve_entity_input(ctx, head)
    relation_id = resolve_relation_input(ctx, relation)

    predicted = False
    prediction_gap = None
    if tail.strip():
        tail_id = resolve_entity_input(ctx, tail)
        prediction_score = ctx.kge.score(head_id, relation_id, tail_id)
    else:
        ranked = ctx.kge.rank_tails(head_id, relation_id, top_k=max(2, predict_top_k), exclude_known=False)
        if not ranked:
            raise ValueError("No tail predictions available for the given head and relation.")
        top = ranked[0]
        tail_id = int(top["tail_id"])
        prediction_score = float(top["score"])
        predicted = True
        if len(ranked) > 1:
            prediction_gap = float(ranked[0]["score"]) - float(ranked[1]["score"])

    subgraph = extract_local_subgraph(ctx, head_id, tail_id)
    paths = enumerate_two_hop_paths(ctx, head_id, relation_id, tail_id, top_k=top_k_paths)
    shared_neighbors = collect_shared_neighbors(ctx, head_id, relation_id, tail_id, top_k=top_k_shared)
    analogies = collect_analogical_support(ctx, head_id, relation_id, tail_id, top_k=top_k_analogies)
    summary = build_summary(ctx, head_id, relation_id, tail_id, paths, shared_neighbors, analogies)

    return {
        "query": {
            "head": ctx.kge.display(ctx.kge.id_to_entity[head_id]),
            "relation": ctx.kge.display(ctx.kge.id_to_relation[relation_id]),
            "tail": ctx.kge.display(ctx.kge.id_to_entity[tail_id]),
        },
        "prediction": {
            "predicted_tail": predicted,
            "score": prediction_score,
            "confidence_gap": prediction_gap,
        },
        "local_subgraph": subgraph,
        "supporting_paths": paths,
        "shared_neighbors": shared_neighbors,
        "similar_entity_support": analogies,
        "summary": summary,
    }
