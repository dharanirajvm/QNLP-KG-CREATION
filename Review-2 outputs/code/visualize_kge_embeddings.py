import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch

from training_fb15k237 import QuantumKGE, setup_quantum


# Replace this with your final trained model snapshot path once ready.
SNAPSHOT_DIR = Path(
    r"C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\inference_snapshots\quantum_fb15k237_20260308_174529_updated_20260310_193344"
)

# Keep in sync with your FB15k-237 split files.
DATASET_DIR = Path(
    r"c:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\datasets\fb15k237"
)

OUT_DIR = SNAPSHOT_DIR / "embedding_viz"
RANDOM_SEED = 42
MAX_ENTITIES_FOR_TSNE = 2000


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
    triples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, r, t = parse_kg_line(line)
        if h in entity_to_id and t in entity_to_id and r in relation_to_id:
            triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
    return triples


def relation_phrase(rel_raw: str, labels: dict[str, str]) -> str:
    if rel_raw in labels:
        return labels[rel_raw]
    clean = rel_raw.strip("/").replace("/", " ").replace("_", " ")
    return clean


def extract_entity_embeddings(
    model: QuantumKGE,
    entity_ids: list[int],
) -> np.ndarray:
    vectors = []
    with torch.no_grad():
        for eid in entity_ids:
            state = model.entity_state(eid).cpu().numpy()
            # Use both real and imaginary parts as final classical embedding features.
            vec = np.concatenate([state.real, state.imag], axis=0)
            vectors.append(vec)
    return np.asarray(vectors, dtype=np.float32)


def build_entity_relation_labels(
    triples: list[tuple[int, int, int]],
    relation_id_to_raw: dict[int, str],
    labels_human: dict[str, str],
) -> dict[int, str]:
    # For each entity, assign the most frequent incident relation as a coarse semantic tag.
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
    palette = plt.cm.get_cmap("tab20", len(uniq))
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
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_json(SNAPSHOT_DIR / "config.json")
    entity_to_id = load_json(SNAPSHOT_DIR / "entity_to_id.json")
    relation_to_id = load_json(SNAPSHOT_DIR / "relation_to_id.json")
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    labels_human = {}
    labels_path = SNAPSHOT_DIR / "labels_human.json"
    if labels_path.exists():
        labels_human = load_json(labels_path)

    cfg_args = config.get("args", {})
    num_qubits = int(cfg_args.get("num_qubits", 6))
    backend = str(cfg_args.get("q_backend", "default.qubit"))
    num_entities = int(config.get("num_entities", len(entity_to_id)))
    num_relations = int(config.get("num_relations", len(relation_to_id)))

    setup_quantum(num_qubits, backend)
    model = QuantumKGE(num_entities=num_entities, num_relations=num_relations, num_qubits=num_qubits)
    state = torch.load(SNAPSHOT_DIR / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    train_triples = load_train_triples_ids(DATASET_DIR, entity_to_id, relation_to_id)
    entity_tag = build_entity_relation_labels(train_triples, id_to_relation, labels_human)

    all_entity_ids = list(range(num_entities))
    if len(all_entity_ids) > MAX_ENTITIES_FOR_TSNE:
        entity_ids = random.sample(all_entity_ids, MAX_ENTITIES_FOR_TSNE)
    else:
        entity_ids = all_entity_ids

    X = extract_entity_embeddings(model, entity_ids)

    tags = [entity_tag.get(e, "other") for e in entity_ids]
    ent_text = [labels_human.get(id_to_entity[e], id_to_entity[e]) for e in entity_ids]

    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_coords = pca.fit_transform(X)
    plot_scatter(
        coords=pca_coords,
        labels=tags,
        out_path=OUT_DIR / "entity_embeddings_pca.png",
        title="Entity Embeddings (PCA) colored by dominant relation",
    )

    # t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        n_iter=1200,
    )
    tsne_coords = tsne.fit_transform(X)
    plot_scatter(
        coords=tsne_coords,
        labels=tags,
        out_path=OUT_DIR / "entity_embeddings_tsne.png",
        title="Entity Embeddings (t-SNE) colored by dominant relation",
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
    df.to_csv(OUT_DIR / "entity_embeddings_projection.csv", index=False, encoding="utf-8")

    summary = {
        "snapshot_dir": str(SNAPSHOT_DIR),
        "dataset_dir": str(DATASET_DIR),
        "backend": backend,
        "num_qubits": num_qubits,
        "num_entities_total": num_entities,
        "num_entities_visualized": len(entity_ids),
        "outputs": [
            str(OUT_DIR / "entity_embeddings_pca.png"),
            str(OUT_DIR / "entity_embeddings_tsne.png"),
            str(OUT_DIR / "entity_embeddings_projection.csv"),
        ],
    }
    (OUT_DIR / "viz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Visualization generated:")
    for x in summary["outputs"]:
        print(" -", x)


if __name__ == "__main__":
    main()

