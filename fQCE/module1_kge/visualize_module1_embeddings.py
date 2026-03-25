#!/usr/bin/env python3
"""Create embedding visualizations for a trained module1 KGE snapshot."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

THIS_DIR = Path(__file__).resolve().parent
FQCE_DIR = THIS_DIR.parent
if str(FQCE_DIR) not in sys.path:
    sys.path.insert(0, str(FQCE_DIR))

from training_fb15k237 import ComplexKGE, QuantumKGE, setup_quantum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize KGE entity embeddings with PCA and t-SNE.")
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Training run directory with model + maps.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing train/valid/test txt files.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <snapshot-dir>/embedding_viz).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-entities-for-tsne", type=int, default=2000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iters", type=int, default=1200)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_kg_line(line: str) -> tuple[str, str, str]:
    parts = line.strip().split("\t") if "\t" in line else line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid triple line: {line}")
    return parts[0], parts[1], parts[2]


def load_train_triples_ids(
    dataset_dir: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> list[tuple[int, int, int]]:
    path = dataset_dir / "train.txt"
    triples: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, r, t = parse_kg_line(line)
        if h in entity_to_id and r in relation_to_id and t in entity_to_id:
            triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    return triples


def relation_phrase(rel_raw: str, labels: dict[str, str]) -> str:
    if rel_raw in labels:
        return labels[rel_raw]
    return rel_raw.strip("/").replace("/", " ").replace("_", " ")


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


def extract_entity_embeddings(
    model_name: str,
    model,
    entity_ids: list[int],
) -> np.ndarray:
    if model_name == "complex":
        emb = []
        with torch.no_grad():
            for eid in entity_ids:
                vec = torch.cat(
                    [model.ent_re.weight[eid], model.ent_im.weight[eid]],
                    dim=0,
                ).cpu().numpy()
                emb.append(vec)
        return np.asarray(emb, dtype=np.float32)

    vectors = []
    with torch.no_grad():
        for eid in entity_ids:
            state = model.entity_state(eid).cpu().numpy()
            vec = np.concatenate([state.real, state.imag], axis=0)
            vectors.append(vec)
    return np.asarray(vectors, dtype=np.float32)


def build_entity_relation_labels(
    triples: list[tuple[int, int, int]],
    relation_id_to_raw: dict[int, str],
    labels_human: dict[str, str],
) -> dict[int, str]:
    rel_counts_by_entity: dict[int, Counter] = defaultdict(Counter)
    for h, r, t in triples:
        rel_counts_by_entity[h][r] += 1
        rel_counts_by_entity[t][r] += 1

    entity_to_tag: dict[int, str] = {}
    for e, ctr in rel_counts_by_entity.items():
        top_r, _ = ctr.most_common(1)[0]
        entity_to_tag[e] = relation_phrase(relation_id_to_raw[top_r], labels_human)
    return entity_to_tag


def plot_scatter(
    coords: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
    max_legend_items: int = 20,
) -> None:
    plt.figure(figsize=(12, 8))

    uniq = sorted(set(labels))
    palette = plt.get_cmap("tab20", len(uniq))
    label_to_color = {lab: palette(i) for i, lab in enumerate(uniq)}

    for lab in uniq:
        idx = [i for i, x in enumerate(labels) if x == lab]
        pts = coords[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7, c=[label_to_color[lab]], label=lab)

    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    if len(uniq) <= max_legend_items:
        plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    snapshot_dir = args.snapshot_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (snapshot_dir / "embedding_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(snapshot_dir / "config.json")
    entity_to_id = load_json(snapshot_dir / "entity_to_id.json")
    relation_to_id = load_json(snapshot_dir / "relation_to_id.json")
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    labels_human = {}
    labels_path = snapshot_dir / "labels_human.json"
    if labels_path.exists():
        labels_human = load_json(labels_path)

    num_entities = int(config.get("num_entities", len(entity_to_id)))
    num_relations = int(config.get("num_relations", len(relation_to_id)))

    model_name, model = load_model(snapshot_dir, config, num_entities, num_relations)

    train_triples = load_train_triples_ids(dataset_dir, entity_to_id, relation_to_id)
    entity_tag = build_entity_relation_labels(train_triples, id_to_relation, labels_human)

    all_entity_ids = list(range(num_entities))
    if len(all_entity_ids) > args.max_entities_for_tsne:
        entity_ids = random.sample(all_entity_ids, args.max_entities_for_tsne)
    else:
        entity_ids = all_entity_ids

    if len(entity_ids) < 2:
        raise ValueError("Need at least 2 entities for projection plots.")

    X = extract_entity_embeddings(model_name, model, entity_ids)
    tags = [entity_tag.get(e, "other") for e in entity_ids]
    ent_text = [labels_human.get(id_to_entity[e], id_to_entity[e]) for e in entity_ids]

    pca = PCA(n_components=2, random_state=args.seed)
    pca_coords = pca.fit_transform(X)
    plot_scatter(
        coords=pca_coords,
        labels=tags,
        out_path=out_dir / "entity_embeddings_pca.png",
        title=f"Entity Embeddings (PCA) - model={model_name}",
    )

    tsne_perplexity = min(float(args.tsne_perplexity), float(max(1, len(entity_ids) - 1)))
    tsne_kwargs = {
        "n_components": 2,
        "random_state": args.seed,
        "perplexity": tsne_perplexity,
        "learning_rate": "auto",
        "init": "pca",
    }
    try:
        tsne = TSNE(max_iter=args.tsne_iters, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=args.tsne_iters, **tsne_kwargs)
    tsne_coords = tsne.fit_transform(X)
    plot_scatter(
        coords=tsne_coords,
        labels=tags,
        out_path=out_dir / "entity_embeddings_tsne.png",
        title=f"Entity Embeddings (t-SNE) - model={model_name}",
    )

    df = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "entity_raw": [id_to_entity[e] for e in entity_ids],
            "entity_text": ent_text,
            "dominant_relation": tags,
            "pca_x": pca_coords[:, 0],
            "pca_y": pca_coords[:, 1],
            "tsne_x": tsne_coords[:, 0],
            "tsne_y": tsne_coords[:, 1],
        }
    )
    df.to_csv(out_dir / "entity_embeddings_projection.csv", index=False, encoding="utf-8")

    summary = {
        "snapshot_dir": str(snapshot_dir),
        "dataset_dir": str(dataset_dir),
        "model": model_name,
        "num_entities_total": num_entities,
        "num_entities_visualized": len(entity_ids),
        "tsne_perplexity_used": tsne_perplexity,
        "outputs": [
            str(out_dir / "entity_embeddings_pca.png"),
            str(out_dir / "entity_embeddings_tsne.png"),
            str(out_dir / "entity_embeddings_projection.csv"),
        ],
    }
    (out_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Visualization generated:")
    for x in summary["outputs"]:
        print(" -", x)


if __name__ == "__main__":
    main()
