import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

from training_fb15k237 import QuantumKGE, setup_quantum


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_kg_line(line: str) -> tuple[str, str, str]:
    parts = line.strip().split("\t") if "\t" in line else line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid triple line: {line}")
    return parts[0], parts[1], parts[2]


def list_snapshot_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def build_truth_maps(
    dataset_dir: str,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]], set[tuple[int, int, int]]]:
    ds = Path(dataset_dir)

    all_true: set[tuple[int, int, int]] = set()
    for split in ("train.txt", "valid.txt", "test.txt"):
        p = ds / split
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, r, t = parse_kg_line(line)
            if h in entity_to_id and t in entity_to_id and r in relation_to_id:
                all_true.add((entity_to_id[h], relation_to_id[r], entity_to_id[t]))

    tails: dict[tuple[int, int], set[int]] = {}
    heads: dict[tuple[int, int], set[int]] = {}
    for h, r, t in all_true:
        tails.setdefault((h, r), set()).add(t)
        heads.setdefault((r, t), set()).add(h)
    return tails, heads, all_true


def resolve_id(text: str, mapping: dict[str, int], labels: dict[str, str], kind: str) -> int:
    s = text.strip()
    if not s:
        raise ValueError(f"Missing {kind}")
    if s in mapping:
        return mapping[s]
    if s.lower() in mapping:
        return mapping[s.lower()]
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < len(mapping):
            return idx

    # Optional matching by human-readable label text.
    target = s.lower()
    for raw, lbl in labels.items():
        if lbl.lower() == target and raw in mapping:
            return mapping[raw]
    raise KeyError(f"Unknown {kind}: {text}")


@st.cache_resource(show_spinner=False)
def load_snapshot(snapshot_dir: str, backend: str):
    snap = Path(snapshot_dir)
    cfg = load_json(snap / "config.json")
    entity_to_id = load_json(snap / "entity_to_id.json")
    relation_to_id = load_json(snap / "relation_to_id.json")
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    labels = {}
    labels_file = snap / "labels_human.json"
    if labels_file.exists():
        labels = load_json(labels_file)

    cfg_args = cfg.get("args", {})
    num_qubits = int(cfg_args.get("num_qubits", 6))
    q_backend = backend or str(cfg_args.get("q_backend", "default.qubit"))
    num_entities = int(cfg.get("num_entities", len(entity_to_id)))
    num_relations = int(cfg.get("num_relations", len(relation_to_id)))

    setup_quantum(num_qubits, q_backend)
    model = QuantumKGE(num_entities=num_entities, num_relations=num_relations, num_qubits=num_qubits)
    state = torch.load(snap / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "id_to_entity": id_to_entity,
        "id_to_relation": id_to_relation,
        "labels": labels,
        "backend": q_backend,
    }


def pretty(raw: str, labels: dict[str, str]) -> str:
    return labels.get(raw, labels.get(raw.lower(), raw))


def main() -> None:
    st.set_page_config(page_title="KGE Inference UI", layout="wide")
    st.title("Review2 KGE UI (Quantum FB15k-237)")
    st.caption("Choose mode, run inference, and compare predictions with ground truth.")

    root = Path(__file__).resolve().parent
    default_snapshots_root = root / "runs_kge_fb15k237"
    snapshots = list_snapshot_dirs(default_snapshots_root)

    with st.sidebar:
        st.header("Model")
        if snapshots:
            snapshot_dir = st.selectbox("Snapshot / Run directory", [str(p) for p in snapshots], index=0)
        else:
            snapshot_dir = st.text_input("Snapshot / Run directory", value=str(default_snapshots_root))
        backend = st.text_input("Backend override (optional)", value="")

        st.header("Data")
        dataset_dir = st.text_input("Dataset directory (for ground truth)", value=str(root / "datasets" / "fb15k237"))
        exclude_known = st.checkbox("Exclude known true triples from ranking", value=True)

        st.header("Ranking")
        top_k = st.slider("Top-K", 1, 50, 10)
        sample_candidates = st.slider("Candidate sample size (0 = all, slower)", 0, 5000, 512, step=64)
        seed = st.number_input("Seed", value=42, step=1)

    try:
        ctx = load_snapshot(snapshot_dir, backend)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load snapshot: {exc}")
        return

    model = ctx["model"]
    entity_to_id = ctx["entity_to_id"]
    relation_to_id = ctx["relation_to_id"]
    id_to_entity = ctx["id_to_entity"]
    id_to_relation = ctx["id_to_relation"]
    labels = ctx["labels"]

    try:
        tails_truth, heads_truth, all_true = build_truth_maps(dataset_dir, entity_to_id, relation_to_id)
        truth_available = True
    except Exception:
        tails_truth, heads_truth, all_true = {}, {}, set()
        truth_available = False

    st.info(f"Loaded backend: `{ctx['backend']}` | entities={len(entity_to_id)} relations={len(relation_to_id)}")
    if not truth_available:
        st.warning("Ground truth dataset not loaded. Check dataset directory path.")

    mode = st.radio("Mode", ["tail", "head", "score"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        head_in = st.text_input("Head", value="")
    with col2:
        relation_in = st.text_input("Relation", value="")
    with col3:
        tail_in = st.text_input("Tail", value="")

    if st.button("Run Inference", type="primary"):
        if not relation_in.strip():
            st.error("Relation is required.")
            return

        try:
            r_id = resolve_id(relation_in, relation_to_id, labels, "relation")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return

        rng = random.Random(int(seed))
        num_entities = len(entity_to_id)

        if mode == "score":
            try:
                h_id = resolve_id(head_in, entity_to_id, labels, "head")
                t_id = resolve_id(tail_in, entity_to_id, labels, "tail")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return

            with torch.no_grad():
                s = float(model.score(h_id, r_id, t_id).item())

            triple_exists = (h_id, r_id, t_id) in all_true if truth_available else None
            st.subheader("Score Result")
            st.write({
                "head": pretty(id_to_entity[h_id], labels),
                "relation": pretty(id_to_relation[r_id], labels),
                "tail": pretty(id_to_entity[t_id], labels),
                "score": s,
                "ground_truth_exists": triple_exists,
            })
            return

        if mode == "tail":
            try:
                h_id = resolve_id(head_in, entity_to_id, labels, "head")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
            gt = tails_truth.get((h_id, r_id), set()) if truth_available else set()

            cands = list(range(num_entities))
            if sample_candidates > 0 and sample_candidates < num_entities:
                cands = sorted(set(rng.sample(range(num_entities), sample_candidates)) | set(gt))

            filtered = set(gt) if exclude_known else set()
            target_t = None
            if tail_in.strip():
                try:
                    target_t = resolve_id(tail_in, entity_to_id, labels, "tail")
                    filtered.discard(target_t)
                except Exception:
                    target_t = None

            with torch.no_grad():
                sp = model.relation_subject_state(h_id, r_id)
                scored = []
                for c in cands:
                    if c in filtered:
                        continue
                    es = model.entity_state(c)
                    sc = float(torch.real(torch.vdot(es, sp)).item())
                    scored.append((c, sc))
            scored.sort(key=lambda x: x[1], reverse=True)

            rows = []
            for rank, (c, sc) in enumerate(scored[:top_k], start=1):
                rows.append({
                    "rank": rank,
                    "tail": pretty(id_to_entity[c], labels),
                    "score": sc,
                    "is_ground_truth": c in gt,
                })
            st.subheader("Top-K Tail Predictions")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            gt_rows = [{"tail": pretty(id_to_entity[x], labels)} for x in sorted(gt)]
            st.subheader("Ground Truth Tails for (head, relation)")
            st.dataframe(pd.DataFrame(gt_rows) if gt_rows else pd.DataFrame([{"tail": "(none)"}]), use_container_width=True)
            return

        # head mode
        try:
            t_id = resolve_id(tail_in, entity_to_id, labels, "tail")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        gt = heads_truth.get((r_id, t_id), set()) if truth_available else set()

        cands = list(range(num_entities))
        if sample_candidates > 0 and sample_candidates < num_entities:
            cands = sorted(set(rng.sample(range(num_entities), sample_candidates)) | set(gt))

        filtered = set(gt) if exclude_known else set()
        target_h = None
        if head_in.strip():
            try:
                target_h = resolve_id(head_in, entity_to_id, labels, "head")
                filtered.discard(target_h)
            except Exception:
                target_h = None

        with torch.no_grad():
            tail_state = model.entity_state(t_id)
            scored = []
            for c in cands:
                if c in filtered:
                    continue
                sp = model.relation_subject_state(c, r_id)
                sc = float(torch.real(torch.vdot(tail_state, sp)).item())
                scored.append((c, sc))
        scored.sort(key=lambda x: x[1], reverse=True)

        rows = []
        for rank, (c, sc) in enumerate(scored[:top_k], start=1):
            rows.append({
                "rank": rank,
                "head": pretty(id_to_entity[c], labels),
                "score": sc,
                "is_ground_truth": c in gt,
            })
        st.subheader("Top-K Head Predictions")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        gt_rows = [{"head": pretty(id_to_entity[x], labels)} for x in sorted(gt)]
        st.subheader("Ground Truth Heads for (relation, tail)")
        st.dataframe(pd.DataFrame(gt_rows) if gt_rows else pd.DataFrame([{"head": "(none)"}]), use_container_width=True)


if __name__ == "__main__":
    main()
