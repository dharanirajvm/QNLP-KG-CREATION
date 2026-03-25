import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch

from training_fb15k237 import QuantumKGE, setup_quantum

THIS_DIR = Path(__file__).resolve().parent
MODULE1_DIR_CANDIDATES = [
    THIS_DIR / "module1_kge",
    THIS_DIR.parent / "fQCE" / "module1_kge",
    THIS_DIR.parents[1] / "fQCE" / "module1_kge",
]
for candidate in MODULE1_DIR_CANDIDATES:
    if candidate.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from downstream_utils import detect_entities_in_query, detect_relation_from_query, load_context
from task1_kg_completion import run_single as run_kg_completion
from task2_semantic_retrieval import run_semantic_similarity


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_kg_line(line: str) -> tuple[str, str, str]:
    parts = line.strip().split("\t") if "\t" in line else line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid triple line: {line}")
    return parts[0], parts[1], parts[2]


def list_snapshot_dirs(bases: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    seen = set()
    for base in bases:
        if not base.exists():
            continue
        for p in base.iterdir():
            if p.is_dir() and str(p) not in seen:
                dirs.append(p)
                seen.add(str(p))
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)


def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "fQCE").exists() and (p / "Review-2 outputs").exists():
            return p
    return start


def first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p and p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_jsonl_df(path: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    if "epoch_seconds" in df.columns:
        df["cum_hours"] = df["epoch_seconds"].fillna(0).cumsum() / 3600.0
    return df


@st.cache_data(show_spinner=False)
def load_csv_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def render_embedding_section(embedding_viz_dir: str) -> None:
    st.subheader("Embedding Visualizations")
    if not embedding_viz_dir.strip():
        st.info("Embedding visualization path is empty.")
        return

    d = Path(embedding_viz_dir)
    if not d.exists():
        st.warning(f"Embedding viz directory not found: {d}")
        return

    summary_path = d / "viz_summary.json"
    if summary_path.exists():
        try:
            st.json(load_json(summary_path))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read viz summary: {exc}")

    pca_img = d / "entity_embeddings_pca.png"
    tsne_img = d / "entity_embeddings_tsne.png"
    c1, c2 = st.columns(2)
    with c1:
        if pca_img.exists():
            st.image(str(pca_img), caption="Entity Embeddings PCA", use_container_width=True)
        else:
            st.info("PCA image not found.")
    with c2:
        if tsne_img.exists():
            st.image(str(tsne_img), caption="Entity Embeddings t-SNE", use_container_width=True)
        else:
            st.info("t-SNE image not found.")

    projection_csv = d / "entity_embeddings_projection.csv"
    if projection_csv.exists():
        try:
            proj_df = load_csv_df(str(projection_csv))
            st.caption(f"Projection rows: {len(proj_df)}")
            query = st.text_input("Filter projected entities", value="", key="emb_filter")
            if query.strip():
                mask = proj_df["entity_text"].astype(str).str.contains(query, case=False, na=False)
                proj_df = proj_df[mask]
            st.dataframe(proj_df.head(300), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read projection CSV: {exc}")
    else:
        st.info("Projection CSV not found.")


def render_metrics_section(metrics_history_path: str, metrics_viz_dir: str) -> None:
    st.subheader("Training and Evaluation Metrics")
    if metrics_history_path.strip():
        p = Path(metrics_history_path)
        if p.exists():
            try:
                mdf = load_jsonl_df(str(p))
                if not mdf.empty:
                    if {"epoch", "train_loss"}.issubset(mdf.columns):
                        st.line_chart(mdf.set_index("epoch")["train_loss"], height=220)
                    if {"epoch", "train_pair_acc"}.issubset(mdf.columns):
                        st.line_chart(mdf.set_index("epoch")["train_pair_acc"], height=220)
                    eval_cols = [c for c in ["val_mrr", "val_hits@1", "val_hits@3", "val_hits@10"] if c in mdf.columns]
                    if eval_cols:
                        eval_df = mdf[mdf["val_mrr"].notna()] if "val_mrr" in mdf.columns else mdf
                        if not eval_df.empty:
                            st.line_chart(eval_df.set_index("epoch")[eval_cols], height=260)
                    if {"epoch", "epoch_seconds"}.issubset(mdf.columns):
                        time_df = mdf.copy()
                        time_df["epoch_minutes"] = time_df["epoch_seconds"] / 60.0
                        cols = ["epoch_minutes"] + (["cum_hours"] if "cum_hours" in time_df.columns else [])
                        st.line_chart(time_df.set_index("epoch")[cols], height=240)
                    st.dataframe(mdf, use_container_width=True)
                else:
                    st.info("Metrics history file has no records.")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not read metrics history: {exc}")
        else:
            st.warning(f"metrics_history.jsonl not found: {p}")
    else:
        st.info("Metrics history path is empty.")

    if metrics_viz_dir.strip():
        md = Path(metrics_viz_dir)
        if md.exists():
            st.markdown("**Saved Metrics Plots**")
            for img_name in ("training_metrics_overview.png", "eval_metrics_detail.png"):
                ip = md / img_name
                if ip.exists():
                    st.image(str(ip), caption=img_name, use_container_width=True)
            summary_json = md / "metrics_summary_viz.json"
            if summary_json.exists():
                try:
                    st.json(load_json(summary_json))
                except Exception:
                    pass


def render_closest_section(embedding_analysis_dir: str, closest_triples_csv: str) -> None:
    st.subheader("Closest Neighbors and Triples")
    if not embedding_analysis_dir.strip():
        st.info("Embedding analysis path is empty.")
    else:
        d = Path(embedding_analysis_dir)
        if not d.exists():
            st.warning(f"Embedding analysis directory not found: {d}")
        else:
            quality = d / "embedding_quality_report.json"
            if quality.exists():
                try:
                    st.json(load_json(quality))
                except Exception:
                    pass

            ne_csv = d / "nearest_entities.csv"
            if ne_csv.exists():
                try:
                    ne_df = load_csv_df(str(ne_csv))
                    st.markdown("**Nearest Entities**")
                    anchors = ne_df["anchor_text"].dropna().astype(str).unique().tolist()
                    anchors.sort()
                    if anchors:
                        selected_anchor = st.selectbox("Anchor Entity", anchors, key="nearest_anchor")
                        view = ne_df[ne_df["anchor_text"] == selected_anchor].sort_values("rank")
                        st.dataframe(view, use_container_width=True)
                    else:
                        st.dataframe(ne_df.head(200), use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Could not read nearest_entities.csv: {exc}")

            nr_csv = d / "nearest_relations.csv"
            if nr_csv.exists():
                try:
                    nr_df = load_csv_df(str(nr_csv))
                    st.markdown("**Nearest Relations**")
                    rels = nr_df["relation_text"].dropna().astype(str).unique().tolist()
                    rels.sort()
                    if rels:
                        selected_rel = st.selectbox("Anchor Relation", rels, key="nearest_relation")
                        view = nr_df[nr_df["relation_text"] == selected_rel].sort_values("neighbor_rank")
                        st.dataframe(view, use_container_width=True)
                    else:
                        st.dataframe(nr_df.head(200), use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Could not read nearest_relations.csv: {exc}")

    if closest_triples_csv.strip():
        cp = Path(closest_triples_csv)
        if cp.exists():
            try:
                st.markdown("**Closest/Correct Triples CSV**")
                st.dataframe(load_csv_df(str(cp)), use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not read closest triples CSV: {exc}")
        else:
            st.warning(f"Closest triples CSV not found: {cp}")


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


QUESTION_PATTERNS = [
    (re.compile(r"where\s+was\s+(.+?)\s+born\??$", re.I), "/people/person/place_of_birth", "tail"),
    (re.compile(r"what\s+is\s+(.+?)'s\s+profession\??$", re.I), "/people/person/profession", "tail"),
    (re.compile(r"what\s+is\s+the\s+profession\s+of\s+(.+?)\??$", re.I), "/people/person/profession", "tail"),
    (re.compile(r"what\s+is\s+(.+?)'s\s+nationality\??$", re.I), "/people/person/nationality", "tail"),
    (re.compile(r"which\s+country\s+is\s+(.+?)\s+from\??$", re.I), "/people/person/nationality", "tail"),
    (re.compile(r"where\s+did\s+(.+?)\s+study\??$", re.I), "/education/educational_degree/people_with_this_degree./education/education/institution", "tail"),
]


@st.cache_resource(show_spinner=False)
def load_downstream_context(snapshot_dir: str, dataset_dir: str):
    return load_context(snapshot_dir=Path(snapshot_dir), dataset_dir=Path(dataset_dir))


def parse_question(question: str) -> dict:
    for pattern, relation_raw, direction in QUESTION_PATTERNS:
        match = pattern.match(question.strip())
        if match:
            return {"entity_text": match.group(1).strip(), "relation_raw": relation_raw, "direction": direction}
    return {"entity_text": None, "relation_raw": None, "direction": None}


def answer_question_fb15k(ctx, question: str, top_k: int) -> dict:
    parsed = parse_question(question)
    if parsed["entity_text"] is None or parsed["relation_raw"] is None:
        entity_hits = detect_entities_in_query(question, ctx, limit=1)
        relation_raw = detect_relation_from_query(question, ctx.relation_to_id)
        if not entity_hits or relation_raw is None:
            raise ValueError("Could not parse question into a FB15k KG query.")
        parsed = {
            "entity_text": ctx.display(ctx.id_to_entity[entity_hits[0]]),
            "relation_raw": relation_raw,
            "direction": "tail",
        }

    entity_id = ctx.resolve_entity(parsed["entity_text"])
    relation_id = ctx.resolve_relation(parsed["relation_raw"])
    direction = parsed["direction"]

    if direction == "head":
        known_answers = [
            {"answer": ctx.display(ctx.id_to_entity[h]), "sentence": ctx.sentence_for_ids(h, relation_id, entity_id), "source": "known_kg"}
            for h, _, _ in ctx.known_answers(relation_id=relation_id, tail_id=entity_id)
        ]
        ranked = ctx.rank_heads(relation_id, entity_id, top_k=ctx.num_entities, exclude_known=False)
    else:
        known_answers = [
            {"answer": ctx.display(ctx.id_to_entity[t]), "sentence": ctx.sentence_for_ids(entity_id, relation_id, t), "source": "known_kg"}
            for _, _, t in ctx.known_answers(head_id=entity_id, relation_id=relation_id)
        ]
        ranked = ctx.rank_tails(entity_id, relation_id, top_k=ctx.num_entities, exclude_known=False)

    predicted_answers = []
    for row in ranked:
        predicted_answers.append(
            {
                "answer": row["head"] if direction == "head" else row["tail"],
                "sentence": row["sentence"],
                "score": row["score"],
                "source": "kge_completion",
            }
        )

    merged = []
    seen = set()
    for item in known_answers + predicted_answers:
        key = item["answer"].lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    return {
        "question": question,
        "parsed_query": {
            "head": ctx.display(ctx.id_to_entity[entity_id]) if direction != "head" else "?",
            "relation": ctx.display(ctx.id_to_relation[relation_id]),
            "tail": ctx.display(ctx.id_to_entity[entity_id]) if direction == "head" else "?",
        },
        "answers": merged[:top_k],
    }


def render_downstream_tasks_tab(snapshot_dir: str, dataset_dir: str, top_k: int) -> None:
    st.subheader("Downstream Tasks")
    st.caption("Runs on the selected FB15k snapshot and dataset without changing the existing UI behavior.")

    task_kg, task_similarity, task_qa = st.tabs(["KG Completion", "Semantic Similarity", "QA"])

    with task_kg:
        kg_mode = st.radio("Completion Mode", ["tail", "head", "relation"], horizontal=True, key="downstream_kg_mode")
        kg_col1, kg_col2, kg_col3 = st.columns(3)
        with kg_col1:
            kg_head = st.text_input("Head", value="", key="downstream_kg_head")
        with kg_col2:
            kg_relation = st.text_input("Relation", value="", key="downstream_kg_relation")
        with kg_col3:
            kg_tail = st.text_input("Tail", value="", key="downstream_kg_tail")
        include_known = st.checkbox("Include known triples in KG completion results", value=False, key="downstream_include_known")

        if st.button("Run KG Completion", key="run_downstream_kg", type="primary"):
            try:
                with st.spinner("Running KG completion over the selected FB15k model..."):
                    ctx = load_downstream_context(snapshot_dir, dataset_dir)
                    result = run_kg_completion(
                        ctx=ctx,
                        mode=kg_mode,
                        head=kg_head,
                        relation=kg_relation,
                        tail=kg_tail,
                        top_k=top_k,
                        include_known=include_known,
                    )
                st.write(result["query"])
                known_df = pd.DataFrame(result["known_answers"]) if result["known_answers"] else pd.DataFrame([{"info": "(none)"}])
                st.markdown("**Known Answers**")
                st.dataframe(known_df, use_container_width=True)
                pred_df = pd.DataFrame(result["predictions"])
                st.markdown("**Predictions**")
                st.dataframe(pred_df, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with task_similarity:
        st.caption("Computes cosine similarity directly between normalized KGE entity embeddings.")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            anchor = st.text_input("Anchor Entity", value="", key="downstream_similarity_anchor")
        with sim_col2:
            target = st.text_input("Target Entity (optional)", value="", key="downstream_similarity_target")
        include_self = st.checkbox("Include anchor entity in neighbor list", value=False, key="downstream_similarity_include_self")

        if st.button("Run Semantic Similarity", key="run_downstream_similarity", type="primary"):
            if not anchor.strip():
                st.error("Anchor entity is required.")
            else:
                try:
                    with st.spinner("Computing cosine similarity in the KGE embedding space..."):
                        ctx = load_downstream_context(snapshot_dir, dataset_dir)
                        result = run_semantic_similarity(
                            ctx=ctx,
                            anchor=anchor,
                            target=target,
                            top_k=top_k,
                            include_self=include_self,
                        )
                    summary = {"anchor": result["anchor"]}
                    if "target" in result:
                        summary["target"] = result["target"]
                        summary["cosine_similarity"] = result["cosine_similarity"]
                    st.write(summary)
                    st.dataframe(pd.DataFrame(result["neighbors"]), use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

    with task_qa:
        question = st.text_input("Question", value="", key="downstream_question")
        if st.button("Run QA", key="run_downstream_qa", type="primary"):
            if not question.strip():
                st.error("Question is required.")
            else:
                try:
                    with st.spinner("Answering question with exact KG lookup + KGE fallback..."):
                        ctx = load_downstream_context(snapshot_dir, dataset_dir)
                        result = answer_question_fb15k(ctx, question, top_k)
                    st.write(result["parsed_query"])
                    st.dataframe(pd.DataFrame(result["answers"]), use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="KGE Inference UI", layout="wide")
    st.title("Review2 KGE UI (Quantum FB15k-237)")
    st.caption("Choose mode, run inference, and compare predictions with ground truth.")

    root = Path(__file__).resolve().parent
    project_root = find_project_root(root)
    default_snapshots_roots = [
        root / "inference_snapshots",
        root / "runs_kge_fb15k237",
        project_root / "fQCE" / "inference_snapshots",
        project_root / "fQCE" / "runs_kge_fb15k237",
    ]
    snapshots = list_snapshot_dirs(default_snapshots_roots)

    with st.sidebar:
        st.header("Model")
        if snapshots:
            snapshot_dir = st.selectbox("Snapshot / Run directory", [str(p) for p in snapshots], index=0)
        else:
            snapshot_dir = st.text_input(
                "Snapshot / Run directory",
                value=str(default_snapshots_roots[0]),
            )
        backend = st.text_input("Backend override (optional)", value="")

        st.header("Data")
        dataset_dir = st.text_input("Dataset directory (for ground truth)", value=str(root / "datasets" / "fb15k237"))
        exclude_known = st.checkbox("Exclude known true triples from ranking", value=True)

        st.header("Ranking")
        top_k = st.slider("Top-K", 1, 50, 10)
        sample_candidates = st.slider("Candidate sample size (0 = all, slower)", 0, 5000, 512, step=64)
        seed = st.number_input("Seed", value=42, step=1)

        snap_path = Path(snapshot_dir)
        st.header("Artifacts")
        default_embedding_viz_dir = first_existing(
            [
                snap_path / "embedding_viz",
                project_root / "Review-2 outputs" / "outputs" / "embedding_viz",
            ]
        )
        default_embedding_analysis_dir = first_existing(
            [
                snap_path / "embedding_analysis",
                project_root / "Review-2 outputs" / "outputs" / "embedding_analysis",
            ]
        )
        default_metrics_viz_dir = first_existing(
            [
                snap_path / "metrics_viz",
                project_root / "Review-2 outputs" / "outputs" / "metrics_viz",
            ]
        )

        embedding_viz_dir = st.text_input(
            "Embedding Viz Dir",
            value=str(default_embedding_viz_dir) if default_embedding_viz_dir else "",
        )
        embedding_analysis_dir = st.text_input(
            "Embedding Analysis Dir",
            value=str(default_embedding_analysis_dir) if default_embedding_analysis_dir else "",
        )
        metrics_history_path = st.text_input(
            "Metrics History JSONL",
            value=str(snap_path / "metrics_history.jsonl"),
        )
        metrics_viz_dir = st.text_input(
            "Metrics Viz Dir",
            value=str(default_metrics_viz_dir) if default_metrics_viz_dir else "",
        )
        closest_triples_csv = st.text_input(
            "Closest Triples CSV (optional)",
            value=str(snap_path / "rank_correct_valid_top3.csv" if (snap_path / "rank_correct_valid_top3.csv").exists() else ""),
        )

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

    section_infer, section_downstream, section_embed, section_metrics, section_closest = st.tabs(
        ["Inference", "Downstream Tasks", "Embeddings", "Metrics", "Closest Triples"]
    )

    with section_downstream:
        render_downstream_tasks_tab(snapshot_dir, dataset_dir, top_k)

    with section_embed:
        render_embedding_section(embedding_viz_dir)

    with section_metrics:
        render_metrics_section(metrics_history_path, metrics_viz_dir)

    with section_closest:
        render_closest_section(embedding_analysis_dir, closest_triples_csv)

    with section_infer:
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
