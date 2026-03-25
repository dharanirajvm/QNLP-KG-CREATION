#!/usr/bin/env python3
"""Analyze semantic structure of module1 KGE embeddings."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
FQCE_DIR = THIS_DIR.parent
if str(FQCE_DIR) not in sys.path:
    sys.path.insert(0, str(FQCE_DIR))

from training_fb15k237 import ComplexKGE, QuantumKGE, setup_quantum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether trained module1 KGE embeddings are meaningful.")
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Path to trained KGE run directory.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to dataset directory with splits.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: <snapshot-dir>/embedding_analysis).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchors", type=int, default=200)
    parser.add_argument("--nearest-pool", type=int, default=3000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max-metric-triples", type=int, default=2000)
    parser.add_argument("--max-prototype-triples", type=int, default=5000)
    return parser.parse_args()


def load_json(path: Path):
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
    triples: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, r, t = parse_kg_line(line)
        if h in entity_to_id and r in relation_to_id and t in entity_to_id:
            triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    return triples


def pretty(raw: str, labels: dict[str, str]) -> str:
    return labels.get(raw, labels.get(raw.lower(), raw))


def load_model(snapshot_dir: Path, config: dict, num_entities: int, num_relations: int):
    cfg_args = config.get("args", {})
    model_name = str(cfg_args.get("model", "quantum"))
    model_path = snapshot_dir / "best_model.pt"
    if not model_path.exists():
        model_path = snapshot_dir / "last_model.pt"
    if not model_path.exists():
        raise FileNotFoundError("Neither best_model.pt nor last_model.pt found in snapshot directory.")

    if model_name == "complex":
        dim = int(cfg_args.get("embedding_dim", 256))
        model = ComplexKGE(num_entities=num_entities, num_relations=num_relations, dim=dim)
    else:
        num_qubits = int(cfg_args.get("num_qubits", 6))
        backend = str(cfg_args.get("q_backend", "default.qubit"))
        setup_quantum(num_qubits, backend)
        model = QuantumKGE(num_entities=num_entities, num_relations=num_relations, num_qubits=num_qubits)
        model_name = "quantum"

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model_name, model


class ModelAdapter:
    def __init__(self, model_name: str, model):
        self.model_name = model_name
        self.model = model

    @torch.no_grad()
    def entity_embedding(self, eid: int) -> torch.Tensor:
        if self.model_name == "complex":
            return torch.cat([self.model.ent_re.weight[eid], self.model.ent_im.weight[eid]], dim=0).float()
        state = self.model.entity_state(eid)
        return torch.cat([state.real, state.imag], dim=0).float()

    @torch.no_grad()
    def score(self, h: int, r: int, t: int) -> float:
        if self.model_name == "complex":
            h_t = torch.tensor([h], dtype=torch.long)
            r_t = torch.tensor([r], dtype=torch.long)
            t_t = torch.tensor([t], dtype=torch.long)
            return float(self.model.score(h_t, r_t, t_t).item())
        return float(self.model.score(h, r, t).item())


@torch.no_grad()
def get_entity_embedding(adapter: ModelAdapter, eid: int, cache: dict[int, torch.Tensor]) -> torch.Tensor:
    if eid in cache:
        return cache[eid]
    vec = adapter.entity_embedding(eid)
    vec = F.normalize(vec.unsqueeze(0), dim=1).squeeze(0)
    cache[eid] = vec
    return vec


@torch.no_grad()
def get_embeddings_for_ids(adapter: ModelAdapter, ids: list[int], cache: dict[int, torch.Tensor]) -> torch.Tensor:
    vecs = [get_entity_embedding(adapter, eid, cache) for eid in ids]
    return torch.stack(vecs, dim=0)


def compute_nearest_entities(
    adapter: ModelAdapter,
    cache: dict[int, torch.Tensor],
    candidate_ids: list[int],
    id_to_entity: dict[int, str],
    labels: dict[str, str],
    anchors: list[int],
    topk: int = 10,
) -> pd.DataFrame:
    rows = []
    pool_emb = get_embeddings_for_ids(adapter, candidate_ids, cache)
    idx_map = {eid: i for i, eid in enumerate(candidate_ids)}
    for anchor in anchors:
        if anchor not in idx_map:
            continue
        anchor_vec = pool_emb[idx_map[anchor]]
        sims = torch.mv(pool_emb, anchor_vec)
        vals, idx = torch.topk(sims, k=min(topk + 1, len(candidate_ids)))
        rank_out = 0
        for score, j_idx in zip(vals.tolist(), idx.tolist()):
            neighbor = candidate_ids[j_idx]
            if neighbor == anchor:
                continue
            rank_out += 1
            rows.append(
                {
                    "anchor_id": anchor,
                    "anchor_raw": id_to_entity[anchor],
                    "anchor_text": pretty(id_to_entity[anchor], labels),
                    "rank": rank_out,
                    "neighbor_id": neighbor,
                    "neighbor_raw": id_to_entity[neighbor],
                    "neighbor_text": pretty(id_to_entity[neighbor], labels),
                    "cosine_similarity": score,
                }
            )
            if rank_out >= topk:
                break
    return pd.DataFrame(rows)


def relation_prototypes(
    adapter: ModelAdapter,
    cache: dict[int, torch.Tensor],
    triples: list[tuple[int, int, int]],
    num_relations: int,
) -> tuple[torch.Tensor, dict[int, int]]:
    buckets: list[list[torch.Tensor]] = [[] for _ in range(num_relations)]
    for h, r, t in triples:
        h_emb = get_entity_embedding(adapter, h, cache)
        t_emb = get_entity_embedding(adapter, t, cache)
        buckets[r].append(t_emb - h_emb)

    protos = []
    sizes: dict[int, int] = {}
    sample_dim = get_entity_embedding(adapter, 0, cache).shape[0]
    for rel_id in range(num_relations):
        if buckets[rel_id]:
            mat = torch.stack(buckets[rel_id], dim=0)
            proto = mat.mean(dim=0)
            sizes[rel_id] = mat.shape[0]
        else:
            proto = torch.zeros(sample_dim)
            sizes[rel_id] = 0
        protos.append(proto)
    prot = torch.stack(protos, dim=0)
    prot = F.normalize(prot, dim=1)
    return prot, sizes


@torch.no_grad()
def relation_retrieval_accuracy(
    adapter: ModelAdapter,
    cache: dict[int, torch.Tensor],
    triples: list[tuple[int, int, int]],
    prototypes: torch.Tensor,
    max_samples: int = 10000,
) -> dict[str, float]:
    if len(triples) > max_samples:
        triples = random.sample(triples, max_samples)

    correct = 0
    total = 0
    mrr_sum = 0.0
    for h, r, t in triples:
        h_emb = get_entity_embedding(adapter, h, cache)
        t_emb = get_entity_embedding(adapter, t, cache)
        diff = F.normalize((t_emb - h_emb).unsqueeze(0), dim=1)
        scores = torch.mv(prototypes, diff.squeeze(0))
        order = torch.argsort(scores, descending=True)
        rank = int((order == r).nonzero(as_tuple=True)[0].item()) + 1
        correct += int(rank == 1)
        mrr_sum += 1.0 / rank
        total += 1
    return {
        "top1_relation_retrieval": correct / max(1, total),
        "mrr_relation_retrieval": mrr_sum / max(1, total),
        "samples": float(total),
    }


@torch.no_grad()
def score_separation(
    adapter: ModelAdapter,
    triples: list[tuple[int, int, int]],
    num_entities: int,
    max_samples: int = 10000,
) -> dict[str, float]:
    if len(triples) > max_samples:
        triples = random.sample(triples, max_samples)

    pos_scores = []
    neg_scores = []
    for h, r, t in triples:
        pos = adapter.score(h, r, t)
        neg_t = random.randint(0, num_entities - 1)
        while neg_t == t:
            neg_t = random.randint(0, num_entities - 1)
        neg = adapter.score(h, r, neg_t)
        pos_scores.append(pos)
        neg_scores.append(neg)

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)
    return {
        "pos_score_mean": float(pos_arr.mean()) if len(pos_arr) else 0.0,
        "pos_score_std": float(pos_arr.std()) if len(pos_arr) else 0.0,
        "neg_score_mean": float(neg_arr.mean()) if len(neg_arr) else 0.0,
        "neg_score_std": float(neg_arr.std()) if len(neg_arr) else 0.0,
        "margin_mean": float((pos_arr - neg_arr).mean()) if len(pos_arr) else 0.0,
        "pairwise_acc": float((pos_arr > neg_arr).mean()) if len(pos_arr) else 0.0,
        "samples": float(len(pos_arr)),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    snapshot_dir = args.snapshot_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (snapshot_dir / "embedding_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_json(snapshot_dir / "config.json")
    entity_to_id = load_json(snapshot_dir / "entity_to_id.json")
    relation_to_id = load_json(snapshot_dir / "relation_to_id.json")
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    labels = {}
    labels_file = snapshot_dir / "labels_human.json"
    if labels_file.exists():
        labels = load_json(labels_file)

    num_entities = int(cfg.get("num_entities", len(entity_to_id)))
    num_relations = int(cfg.get("num_relations", len(relation_to_id)))

    model_name, model = load_model(snapshot_dir, cfg, num_entities, num_relations)
    adapter = ModelAdapter(model_name=model_name, model=model)

    train_triples = load_split_ids(dataset_dir, "train", entity_to_id, relation_to_id)
    valid_triples = load_split_ids(dataset_dir, "valid", entity_to_id, relation_to_id)
    if len(train_triples) > args.max_prototype_triples:
        train_triples = random.sample(train_triples, args.max_prototype_triples)
    if len(valid_triples) > args.max_metric_triples:
        valid_triples = random.sample(valid_triples, args.max_metric_triples)

    emb_cache: dict[int, torch.Tensor] = {}

    nearest_pool = min(args.nearest_pool, num_entities)
    candidate_ids = random.sample(range(num_entities), nearest_pool)
    anchors = random.sample(candidate_ids, min(args.anchors, len(candidate_ids)))
    nearest_df = compute_nearest_entities(
        adapter=adapter,
        cache=emb_cache,
        candidate_ids=candidate_ids,
        id_to_entity=id_to_entity,
        labels=labels,
        anchors=anchors,
        topk=args.topk,
    )
    nearest_df.to_csv(out_dir / "nearest_entities.csv", index=False, encoding="utf-8")

    prototypes, rel_sizes = relation_prototypes(adapter, emb_cache, train_triples, num_relations)
    rel_rows = []
    rel_sim = prototypes @ prototypes.T
    for rel_id in range(num_relations):
        vals, idx = torch.topk(rel_sim[rel_id], k=min(args.topk + 1, num_relations))
        rank = 0
        for score, nbr in zip(vals.tolist(), idx.tolist()):
            if nbr == rel_id:
                continue
            rank += 1
            rel_rows.append(
                {
                    "relation_id": rel_id,
                    "relation_raw": id_to_relation[rel_id],
                    "relation_text": pretty(id_to_relation[rel_id], labels),
                    "neighbor_rank": rank,
                    "neighbor_relation_id": nbr,
                    "neighbor_relation_raw": id_to_relation[nbr],
                    "neighbor_relation_text": pretty(id_to_relation[nbr], labels),
                    "cosine_similarity": score,
                    "relation_train_count": rel_sizes.get(rel_id, 0),
                }
            )
            if rank >= args.topk:
                break
    pd.DataFrame(rel_rows).to_csv(out_dir / "nearest_relations.csv", index=False, encoding="utf-8")

    rel_acc = relation_retrieval_accuracy(
        adapter=adapter,
        cache=emb_cache,
        triples=valid_triples,
        prototypes=prototypes,
        max_samples=min(args.max_metric_triples, len(valid_triples)),
    )
    separation = score_separation(
        adapter=adapter,
        triples=valid_triples,
        num_entities=num_entities,
        max_samples=min(args.max_metric_triples, len(valid_triples)),
    )

    report = {
        "snapshot_dir": str(snapshot_dir),
        "dataset_dir": str(dataset_dir),
        "model": model_name,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "eval_settings": {
            "anchors": args.anchors,
            "nearest_pool": nearest_pool,
            "topk": args.topk,
            "max_metric_triples": args.max_metric_triples,
            "max_prototype_triples": args.max_prototype_triples,
            "cached_entities_used": len(emb_cache),
        },
        "relation_retrieval": rel_acc,
        "score_separation": separation,
        "artifacts": {
            "nearest_entities_csv": str(out_dir / "nearest_entities.csv"),
            "nearest_relations_csv": str(out_dir / "nearest_relations.csv"),
        },
    }
    (out_dir / "embedding_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Done. Wrote:")
    print(" -", out_dir / "nearest_entities.csv")
    print(" -", out_dir / "nearest_relations.csv")
    print(" -", out_dir / "embedding_quality_report.json")


if __name__ == "__main__":
    main()

