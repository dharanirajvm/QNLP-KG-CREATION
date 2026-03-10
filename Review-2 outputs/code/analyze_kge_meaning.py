import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from training_fb15k237 import QuantumKGE, setup_quantum


# Replace with your final snapshot when ready.
SNAPSHOT_DIR = Path(
    r"c:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\inference_snapshots\quantum_fb15k237_20260308_174529_updated_20260310_193344"
)
DATASET_DIR = Path(
    r"c:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\datasets\fb15k237"
)
OUT_DIR = SNAPSHOT_DIR / "embedding_analysis"

SEED = 42
MAX_ENTITY_NEIGHBORS_ANCHORS = 200
MAX_TRIPLES_FOR_METRICS = 2000
MAX_PROTOTYPE_TRAIN_TRIPLES = 5000
NEAREST_ENTITY_POOL = 3000
TOPK = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether trained quantum KG embeddings are meaningful.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SNAPSHOT_DIR,
        help="Path to trained model snapshot. Replace default with your final model snapshot.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to FB15k-237 dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--anchors", type=int, default=MAX_ENTITY_NEIGHBORS_ANCHORS)
    parser.add_argument("--nearest-pool", type=int, default=NEAREST_ENTITY_POOL)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--max-metric-triples", type=int, default=MAX_TRIPLES_FOR_METRICS)
    parser.add_argument("--max-prototype-triples", type=int, default=MAX_PROTOTYPE_TRAIN_TRIPLES)
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
    triples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, r, t = parse_kg_line(line)
        if h in entity_to_id and r in relation_to_id and t in entity_to_id:
            triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    return triples


def pretty(raw: str, labels: dict[str, str]) -> str:
    return labels.get(raw, labels.get(raw.lower(), raw))


@torch.no_grad()
def get_entity_embedding(model: QuantumKGE, eid: int, cache: dict[int, torch.Tensor]) -> torch.Tensor:
    if eid in cache:
        return cache[eid]
    state = model.entity_state(eid)
    vec = torch.cat([state.real, state.imag], dim=0).float()
    vec = F.normalize(vec.unsqueeze(0), dim=1).squeeze(0)
    cache[eid] = vec
    return vec


@torch.no_grad()
def get_embeddings_for_ids(model: QuantumKGE, ids: list[int], cache: dict[int, torch.Tensor]) -> torch.Tensor:
    vecs = [get_entity_embedding(model, eid, cache) for eid in ids]
    return torch.stack(vecs, dim=0)


def compute_nearest_entities(
    model: QuantumKGE,
    cache: dict[int, torch.Tensor],
    candidate_ids: list[int],
    id_to_entity: dict[int, str],
    labels: dict[str, str],
    anchors: list[int],
    topk: int = 10,
) -> pd.DataFrame:
    rows = []
    pool_emb = get_embeddings_for_ids(model, candidate_ids, cache)
    idx_map = {eid: i for i, eid in enumerate(candidate_ids)}
    for a in anchors:
        if a not in idx_map:
            continue
        a_vec = pool_emb[idx_map[a]]
        sims = torch.mv(pool_emb, a_vec)
        vals, idx = torch.topk(sims, k=min(topk + 1, len(candidate_ids)))
        rank_out = 0
        for score, j_idx in zip(vals.tolist(), idx.tolist()):
            j = candidate_ids[j_idx]
            if j == a:
                continue
            rank_out += 1
            rows.append(
                {
                    "anchor_id": a,
                    "anchor_raw": id_to_entity[a],
                    "anchor_text": pretty(id_to_entity[a], labels),
                    "rank": rank_out,
                    "neighbor_id": j,
                    "neighbor_raw": id_to_entity[j],
                    "neighbor_text": pretty(id_to_entity[j], labels),
                    "cosine_similarity": score,
                }
            )
            if rank_out >= topk:
                break
    return pd.DataFrame(rows)


def relation_prototypes(
    model: QuantumKGE,
    cache: dict[int, torch.Tensor],
    triples: list[tuple[int, int, int]],
    num_relations: int,
) -> tuple[torch.Tensor, dict[int, int]]:
    buckets: list[list[torch.Tensor]] = [[] for _ in range(num_relations)]
    for h, r, t in triples:
        h_emb = get_entity_embedding(model, h, cache)
        t_emb = get_entity_embedding(model, t, cache)
        buckets[r].append(t_emb - h_emb)
    protos = []
    sizes: dict[int, int] = {}
    sample_dim = get_entity_embedding(model, 0, cache).shape[0]
    for r in range(num_relations):
        if buckets[r]:
            mat = torch.stack(buckets[r], dim=0)
            proto = mat.mean(dim=0)
            sizes[r] = mat.shape[0]
        else:
            proto = torch.zeros(sample_dim)
            sizes[r] = 0
        protos.append(proto)
    prot = torch.stack(protos, dim=0)
    prot = F.normalize(prot, dim=1)
    return prot, sizes


@torch.no_grad()
def relation_retrieval_accuracy(
    model: QuantumKGE,
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
        h_emb = get_entity_embedding(model, h, cache)
        t_emb = get_entity_embedding(model, t, cache)
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
    model: QuantumKGE,
    triples: list[tuple[int, int, int]],
    num_entities: int,
    max_samples: int = 10000,
) -> dict[str, float]:
    if len(triples) > max_samples:
        triples = random.sample(triples, max_samples)

    pos_scores = []
    neg_scores = []
    for h, r, t in triples:
        pos = float(model.score(h, r, t).item())
        neg_t = random.randint(0, num_entities - 1)
        while neg_t == t:
            neg_t = random.randint(0, num_entities - 1)
        neg = float(model.score(h, r, neg_t).item())
        pos_scores.append(pos)
        neg_scores.append(neg)

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)
    return {
        "pos_score_mean": float(pos_arr.mean()),
        "pos_score_std": float(pos_arr.std()),
        "neg_score_mean": float(neg_arr.mean()),
        "neg_score_std": float(neg_arr.std()),
        "margin_mean": float((pos_arr - neg_arr).mean()),
        "pairwise_acc": float((pos_arr > neg_arr).mean()),
        "samples": float(len(pos_arr)),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    snapshot_dir: Path = args.snapshot_dir
    dataset_dir: Path = args.dataset_dir
    out_dir = snapshot_dir / "embedding_analysis"
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

    cfg_args = cfg.get("args", {})
    num_qubits = int(cfg_args.get("num_qubits", 6))
    backend = str(cfg_args.get("q_backend", "default.qubit"))
    num_entities = int(cfg.get("num_entities", len(entity_to_id)))
    num_relations = int(cfg.get("num_relations", len(relation_to_id)))

    setup_quantum(num_qubits, backend)
    model = QuantumKGE(num_entities=num_entities, num_relations=num_relations, num_qubits=num_qubits)
    state = torch.load(snapshot_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    train_triples = load_split_ids(dataset_dir, "train", entity_to_id, relation_to_id)
    valid_triples = load_split_ids(dataset_dir, "valid", entity_to_id, relation_to_id)
    if len(train_triples) > args.max_prototype_triples:
        train_triples = random.sample(train_triples, args.max_prototype_triples)
    if len(valid_triples) > args.max_metric_triples:
        valid_triples = random.sample(valid_triples, args.max_metric_triples)
    emb_cache: dict[int, torch.Tensor] = {}

    # Nearest entities (human-interpretable examples)
    nearest_pool = min(args.nearest_pool, num_entities)
    candidate_ids = random.sample(range(num_entities), nearest_pool)
    anchors = random.sample(candidate_ids, min(args.anchors, len(candidate_ids)))
    nearest_df = compute_nearest_entities(
        model=model,
        cache=emb_cache,
        candidate_ids=candidate_ids,
        id_to_entity=id_to_entity,
        labels=labels,
        anchors=anchors,
        topk=args.topk,
    )
    nearest_df.to_csv(out_dir / "nearest_entities.csv", index=False, encoding="utf-8")

    # Relation-level structure from translation-like prototypes.
    protos, rel_sizes = relation_prototypes(model, emb_cache, train_triples, num_relations)
    rel_rows = []
    rel_sim = protos @ protos.T
    for r in range(num_relations):
        vals, idx = torch.topk(rel_sim[r], k=min(args.topk + 1, num_relations))
        rank = 0
        for s, j in zip(vals.tolist(), idx.tolist()):
            if j == r:
                continue
            rank += 1
            rel_rows.append(
                {
                    "relation_id": r,
                    "relation_raw": id_to_relation[r],
                    "relation_text": pretty(id_to_relation[r], labels),
                    "neighbor_rank": rank,
                    "neighbor_relation_id": j,
                    "neighbor_relation_raw": id_to_relation[j],
                    "neighbor_relation_text": pretty(id_to_relation[j], labels),
                    "cosine_similarity": s,
                    "relation_train_count": rel_sizes.get(r, 0),
                }
            )
            if rank >= args.topk:
                break
    pd.DataFrame(rel_rows).to_csv(out_dir / "nearest_relations.csv", index=False, encoding="utf-8")

    rel_acc = relation_retrieval_accuracy(
        model=model,
        cache=emb_cache,
        triples=valid_triples,
        prototypes=protos,
        max_samples=min(args.max_metric_triples, len(valid_triples)),
    )
    sep = score_separation(
        model=model,
        triples=valid_triples,
        num_entities=num_entities,
        max_samples=min(args.max_metric_triples, len(valid_triples)),
    )

    report = {
        "snapshot_dir": str(snapshot_dir),
        "dataset_dir": str(dataset_dir),
        "backend": backend,
        "num_qubits": num_qubits,
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
        "score_separation": sep,
        "artifacts": {
            "nearest_entities_csv": str(out_dir / "nearest_entities.csv"),
            "nearest_relations_csv": str(out_dir / "nearest_relations.csv"),
        },
    }
    (out_dir / "embedding_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("Done. Wrote:")
    print(" -", out_dir / "nearest_entities.csv")
    print(" -", out_dir / "nearest_relations.csv")
    print(" -", out_dir / "embedding_quality_report.json")


if __name__ == "__main__":
    main()
