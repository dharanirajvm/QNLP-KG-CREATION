import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

from training_fb15k237 import QuantumKGE, setup_quantum

try:
    import networkx as nx
    from pyvis.network import Network
except Exception:  # noqa: BLE001
    nx = None
    Network = None

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

REVIEW3_CODE_CANDIDATES = [
    THIS_DIR.parent / "Review-3 outputs" / "code",
    THIS_DIR.parents[1] / "Review-3 outputs" / "code",
    THIS_DIR.parents[2] / "Review-3 outputs" / "code",
]
for candidate in REVIEW3_CODE_CANDIDATES:
    if candidate.exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from downstream_utils import detect_entities_in_query, detect_relation_from_query, load_context
from task1_kg_completion import run_single as run_kg_completion
from task2_semantic_retrieval import run_semantic_similarity
from explanation_utils import build_explanation_context, explain_prediction


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


def default_fb15k_dataset_dir(root: Path, project_root: Path) -> Path:
    candidates = [
        root / "datasets" / "fb15k237",
        project_root / "fQCE" / "datasets" / "fb15k237",
        project_root / "datasets" / "fb15k237",
    ]
    existing = first_existing(candidates)
    return existing if existing is not None else candidates[1]


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


@st.cache_resource(show_spinner=False)
def load_explanation_context(snapshot_dir: str, dataset_dir: str):
    return build_explanation_context(snapshot_dir=Path(snapshot_dir), dataset_dir=Path(dataset_dir))


def score_tier(score: float | None, high: float, medium: float) -> tuple[str, str]:
    if score is None:
        return "N/A", "#64748b"
    if score >= high:
        return "High", "#15803d"
    if score >= medium:
        return "Medium", "#b45309"
    return "Low", "#b91c1c"


def metric_card(title: str, value: str, subtitle: str, color: str) -> str:
    return f"""
    <div style="
        border:1px solid rgba(148,163,184,0.35);
        border-left:6px solid {color};
        border-radius:14px;
        padding:16px 18px;
        background:linear-gradient(180deg, rgba(248,250,252,0.98), rgba(241,245,249,0.96));
        box-shadow:0 6px 18px rgba(15,23,42,0.06);
        min-height:126px;">
      <div style="font-size:0.82rem;color:#475569;font-weight:600;letter-spacing:0.02em;">{title}</div>
      <div style="font-size:2rem;font-weight:800;color:{color};margin-top:8px;line-height:1.1;">{value}</div>
      <div style="font-size:0.88rem;color:#334155;margin-top:10px;">{subtitle}</div>
    </div>
    """


def format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def color_for_score(value: float | None, high: float, medium: float) -> str:
    return "color: inherit !important; font-weight: 700;"


def style_score_table(df: pd.DataFrame, score_cols: list[tuple[str, float, float]]):
    styler = df.style
    for col, high, medium in score_cols:
        if col in df.columns:
            styler = styler.map(lambda value: color_for_score(value, high, medium), subset=[col])
            styler = styler.format({col: "{:.4f}"})
    return styler


def present_score_table(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    shown = df.copy()
    for col in score_cols:
        if col in shown.columns:
            shown[col] = shown[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    return shown


def render_explanation_graph_html(subgraph: dict, title: str) -> str:
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])
    if not nodes:
        return "<div style='padding:16px;border:1px dashed #94a3b8;border-radius:12px;'>No local subgraph nodes found.</div>"

    if Network is None or nx is None:
        items = [
            "<div style='padding:12px 0;'>",
            f"<h4 style='margin:0 0 12px 0;color:#0f172a;'>{title}</h4>",
            "<ul style='margin:0;padding-left:18px;'>",
        ]
        for edge in edges[:50]:
            items.append(
                "<li>"
                f"{edge['head']} -- {edge['relation']} -- {edge['tail']}"
                "</li>"
            )
        items.append("</ul></div>")
        return "\n".join(items)

    graph = nx.DiGraph()
    node_map = {node["entity_id"]: node for node in nodes}
    for node in nodes:
        degree = int(node.get("degree", 0))
        graph.add_node(
            node["entity_id"],
            label=node["entity"],
            title=f"{node['entity']}<br>degree={degree}",
            size=max(18, min(42, 12 + degree // 8)),
        )
    for edge in edges:
        graph.add_edge(
            edge["head_id"],
            edge["tail_id"],
            label=edge["relation"],
            title=edge["relation"],
        )

    net = Network(height="640px", width="100%", directed=True, bgcolor="#f8fafc", font_color="#0f172a")
    net.from_nx(graph)
    for node in net.nodes:
        degree = int(node_map.get(node["id"], {}).get("degree", 0))
        node["shape"] = "dot"
        node["color"] = {
            "background": "#0f766e" if degree < 60 else "#1d4ed8",
            "border": "#0f172a",
            "highlight": {"background": "#ea580c", "border": "#7c2d12"},
        }
        node["font"] = {"size": 18, "face": "Georgia"}
    for edge in net.edges:
        edge["arrows"] = "to"
        edge["color"] = {"color": "#64748b", "highlight": "#ea580c"}
        edge["smooth"] = {"enabled": True, "type": "dynamic"}
        edge["font"] = {"size": 13, "align": "top"}
    net.set_options(
        """
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3200,
              "springLength": 160,
              "springConstant": 0.025
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """
    )
    return net.generate_html(notebook=False)


def render_explanation_paths_graph_html(query: dict, paths: list[dict], highlighted_rank: int | None = None) -> str:
    if not paths:
        return "<div style='padding:16px;border:1px dashed #94a3b8;border-radius:12px;'>No ranked explanation paths available for this triple.</div>"

    head = query["head"]
    tail = query["tail"]
    if highlighted_rank is None:
        highlighted_rank = int(paths[0]["rank"])

    if Network is None or nx is None:
        items = [
            "<div style='padding:12px 0;'>",
            "<h4 style='margin:0 0 12px 0;color:#0f172a;'>Top Explanation Paths</h4>",
            "<ul style='margin:0;padding-left:18px;'>",
        ]
        for row in paths:
            marker = " [selected]" if int(row["rank"]) == highlighted_rank else ""
            items.append(
                f"<li>#{row['rank']} {head} -- {row['path_pattern'][0]} -- {row['intermediate']} -- "
                f"{row['path_pattern'][1]} -- {tail}{marker}</li>"
            )
        items.append("</ul></div>")
        return "\n".join(items)

    graph = nx.MultiDiGraph()
    graph.add_node(head, label=head, title="Head entity", size=36, color="#0f766e")
    graph.add_node(tail, label=tail, title="Tail entity", size=36, color="#b91c1c")

    for row in paths:
        rank = int(row["rank"])
        intermediate = row["intermediate"]
        selected = rank == highlighted_rank
        edge_color = "#ea580c" if selected else "#475569"
        edge_width = 5 if selected else 2
        node_color = "#f59e0b" if selected else "#94a3b8"

        graph.add_node(
            intermediate,
            label=intermediate,
            title=(
                f"Path rank {rank}<br>"
                f"Explanation score={float(row['explanation_score']):.4f}<br>"
                f"Relation relevance={float(row['relation_relevance']):.4f}"
            ),
            size=30 if selected else 22,
            color=node_color,
        )
        graph.add_edge(
            head,
            intermediate,
            label=row["path_pattern"][0],
            title=f"Rank {rank} | {row['path_pattern'][0]}",
            color=edge_color,
            width=edge_width,
        )
        graph.add_edge(
            intermediate,
            tail,
            label=row["path_pattern"][1],
            title=f"Rank {rank} | {row['path_pattern'][1]}",
            color=edge_color,
            width=edge_width,
        )

    net = Network(height="620px", width="100%", directed=True, bgcolor="#fffdf8", font_color="#0f172a")
    net.from_nx(graph)
    for node in net.nodes:
        node["shape"] = "dot"
        color = node.get("color", "#94a3b8")
        node["color"] = {
            "background": color,
            "border": "#0f172a",
            "highlight": {"background": "#fb923c", "border": "#7c2d12"},
        }
        node["font"] = {"size": 18, "face": "Georgia"}
    for edge in net.edges:
        color = edge.get("color", "#475569")
        width = edge.get("width", 2)
        edge["arrows"] = "to"
        edge["color"] = {"color": color, "highlight": "#ea580c"}
        edge["width"] = width
        edge["smooth"] = {"enabled": True, "type": "curvedCW"}
        edge["font"] = {"size": 13, "align": "top"}
    net.set_options(
        """
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2800,
              "springLength": 170,
              "springConstant": 0.03
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """
    )
    return net.generate_html(notebook=False)


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


def render_explainability_tab(snapshot_dir: str, dataset_dir: str, top_k: int) -> None:
    st.subheader("Explainability Layer")
    st.caption("Explains a provided or predicted FB15k triple using symbolic paths, shared neighbors, analogical support, and a local graph view.")

    st.markdown(
        """
        <style>
        .exp-chip {
            display:inline-block;
            padding:6px 10px;
            border-radius:999px;
            font-size:0.82rem;
            font-weight:700;
            margin-right:8px;
            margin-bottom:8px;
            border:1px solid rgba(148,163,184,0.45);
            background:#f8fafc;
            color:#0f172a;
        }
        .exp-summary {
            padding:18px 20px;
            border-radius:16px;
            background:linear-gradient(135deg, #fff7ed, #eff6ff);
            border:1px solid rgba(148,163,184,0.35);
            box-shadow:0 8px 24px rgba(15,23,42,0.06);
        }
        .exp-title {
            font-size:0.85rem;
            font-weight:700;
            color:#475569;
            letter-spacing:0.03em;
            text-transform:uppercase;
        }
        .exp-body {
            margin-top:8px;
            font-size:1.05rem;
            color:#0f172a;
            line-height:1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.15, 1.15, 1.0])
    with col1:
        exp_head = st.text_input("Head Entity", value="Steven Spielberg", key="exp_head")
    with col2:
        exp_relation = st.text_input("Relation", value="nationality", key="exp_relation")
    with col3:
        exp_tail = st.text_input("Tail Entity (optional)", value="United States of America", key="exp_tail")

    config1, config2, config3 = st.columns(3)
    with config1:
        predict_top_k = st.slider("Prediction Candidate Depth", 2, 10, 3, key="exp_predict_topk")
    with config2:
        top_k_paths = st.slider("Top Paths", 1, 10, min(5, top_k), key="exp_top_paths")
    with config3:
        top_k_shared = st.slider("Shared / Analogical Rows", 1, 10, min(5, top_k), key="exp_top_shared")

    if st.button("Explain Triple", key="run_explainability", type="primary"):
        try:
            with st.spinner("Building explanation context and ranking evidence..."):
                exp_ctx = load_explanation_context(snapshot_dir, dataset_dir)
                result = explain_prediction(
                    exp_ctx,
                    head=exp_head,
                    relation=exp_relation,
                    tail=exp_tail,
                    predict_top_k=predict_top_k,
                    top_k_paths=top_k_paths,
                    top_k_shared=top_k_shared,
                    top_k_analogies=top_k_shared,
                )

            query = result["query"]
            prediction = result["prediction"]
            paths = result["supporting_paths"]
            shared_neighbors = result["shared_neighbors"]
            analogies = result["similar_entity_support"]

            path_score = float(paths[0]["explanation_score"]) if paths else None
            shared_score = float(shared_neighbors[0]["shared_neighbor_score"]) if shared_neighbors else None
            analogy_score = float(analogies[0]["support_score"]) if analogies else None
            best_evidence_label = "Supporting Paths" if paths else "Shared Neighbors" if shared_neighbors else "Analogical Support" if analogies else "Latent Score Only"

            score_label, score_color = score_tier(float(prediction["score"]), 0.72, 0.58)
            ev_label, ev_color = score_tier(path_score or shared_score or analogy_score, 0.70, 0.45)
            gap_label, gap_color = score_tier(prediction["confidence_gap"], 0.15, 0.05)

            q1, q2, q3 = st.columns(3)
            with q1:
                components.html(
                    metric_card(
                        "Prediction Score",
                        format_score(float(prediction["score"])),
                        f"{score_label} model support for {query['head']} -> {query['tail']}",
                        score_color,
                    ),
                    height=150,
                )
            with q2:
                components.html(
                    metric_card(
                        "Best Evidence Score",
                        format_score(path_score or shared_score or analogy_score),
                        f"{ev_label} evidence from {best_evidence_label}",
                        ev_color,
                    ),
                    height=150,
                )
            with q3:
                components.html(
                    metric_card(
                        "Confidence Gap",
                        format_score(prediction["confidence_gap"]),
                        f"{gap_label} separation from next tail candidate",
                        gap_color,
                    ),
                    height=150,
                )

            chips = [
                f"<span class='exp-chip'>Head: {query['head']}</span>",
                f"<span class='exp-chip'>Relation: {query['relation']}</span>",
                f"<span class='exp-chip'>Tail: {query['tail']}</span>",
                f"<span class='exp-chip'>Tail Source: {'Predicted' if prediction['predicted_tail'] else 'Provided / Known'}</span>",
                f"<span class='exp-chip'>Best Evidence: {best_evidence_label}</span>",
            ]
            st.markdown("".join(chips), unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="exp-summary">
                  <div class="exp-title">Explanation Summary</div>
                  <div class="exp-body">{result['summary']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            selected_rank = None
            if paths:
                path_options = {
                    f"Rank {int(row['rank'])} | score={float(row['explanation_score']):.4f} | via {row['intermediate']}": int(row["rank"])
                    for row in paths
                }
                selected_label = st.selectbox(
                    "Highlighted Explanation Path",
                    list(path_options.keys()),
                    key="exp_selected_path",
                )
                selected_rank = path_options[selected_label]
                selected_path = next(row for row in paths if int(row["rank"]) == selected_rank)
                st.markdown(
                    f"""
                    <div class="exp-summary" style="margin-top:6px;">
                      <div class="exp-title">Selected Path</div>
                      <div class="exp-body">{selected_path['path_sentence']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            graph_col, overview_col = st.columns([1.45, 1.0])
            with graph_col:
                st.markdown("**Top Explanation Paths Graph**")
                path_graph_html = render_explanation_paths_graph_html(query, paths, highlighted_rank=selected_rank)
                components.html(path_graph_html, height=660, scrolling=True)
            with overview_col:
                st.markdown("**Evidence Coverage**")
                cov_rows = [
                    {"evidence_type": "supporting_paths", "count": len(paths)},
                    {"evidence_type": "shared_neighbors", "count": len(shared_neighbors)},
                    {"evidence_type": "analogical_support", "count": len(analogies)},
                    {"evidence_type": "local_nodes", "count": len(result["local_subgraph"].get("nodes", []))},
                    {"evidence_type": "local_edges", "count": len(result["local_subgraph"].get("edges", []))},
                ]
                st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)
                st.markdown("**Query Details**")
                st.json({"query": query, "prediction": prediction})

            st.markdown("**Local Subgraph Context**")
            st.caption("This broader neighborhood view provides context around the entities. It is secondary to the dedicated explanation-path graph above.")
            graph_html = render_explanation_graph_html(
                result["local_subgraph"],
                f"{query['head']} -- {query['relation']} -- {query['tail']}",
            )
            components.html(graph_html, height=680, scrolling=True)

            evidence_paths, evidence_shared, evidence_analogies = st.tabs(
                ["Supporting Paths", "Shared Neighbors", "Analogical Support"]
            )

            with evidence_paths:
                if paths:
                    paths_df = pd.DataFrame(paths)
                    display_cols = [
                        "rank",
                        "intermediate",
                        "path_sentence",
                        "explanation_score",
                        "path_reliability",
                        "relation_relevance",
                        "path_frequency",
                        "embedding_support",
                    ]
                    paths_df = paths_df[[col for col in display_cols if col in paths_df.columns]]
                    st.dataframe(
                        present_score_table(
                            paths_df,
                            [
                                "explanation_score",
                                "path_reliability",
                                "relation_relevance",
                                "embedding_support",
                            ],
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("No two-hop symbolic paths were found for this triple.")

            with evidence_shared:
                if shared_neighbors:
                    shared_df = pd.DataFrame(shared_neighbors)
                    display_cols = [
                        "rank",
                        "neighbor",
                        "shared_neighbor_score",
                        "embedding_coherence",
                        "degree_penalty",
                        "relation_match_strength",
                        "head_to_neighbor_relations",
                        "tail_to_neighbor_relations",
                        "neighbor_to_head_relations",
                        "neighbor_to_tail_relations",
                    ]
                    shared_df = shared_df[[col for col in display_cols if col in shared_df.columns]]
                    st.dataframe(
                        present_score_table(
                            shared_df,
                            [
                                "shared_neighbor_score",
                                "embedding_coherence",
                                "degree_penalty",
                                "relation_match_strength",
                            ],
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("No shared-neighbor structural evidence was found for this triple.")

            with evidence_analogies:
                if analogies:
                    analog_df = pd.DataFrame(analogies)
                    display_cols = [
                        "rank",
                        "support_type",
                        "analog_head",
                        "analog_relation",
                        "analog_tail",
                        "support_score",
                        "head_similarity",
                        "tail_similarity",
                    ]
                    analog_df = analog_df[[col for col in display_cols if col in analog_df.columns]]
                    st.dataframe(
                        present_score_table(
                            analog_df,
                            [
                                "support_score",
                                "head_similarity",
                                "tail_similarity",
                            ],
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("No analogical KGE support rows were found for this triple.")

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
        dataset_dir = st.text_input(
            "Dataset directory (for ground truth)",
            value=str(default_fb15k_dataset_dir(root, project_root)),
        )
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

    section_infer, section_downstream, section_explain, section_embed, section_metrics, section_closest = st.tabs(
        ["Inference", "Downstream Tasks", "Explainability", "Embeddings", "Metrics", "Closest Triples"]
    )

    with section_downstream:
        render_downstream_tasks_tab(snapshot_dir, dataset_dir, top_k)

    with section_explain:
        render_explainability_tab(snapshot_dir, dataset_dir, top_k)

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
