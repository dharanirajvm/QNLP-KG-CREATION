#!/usr/bin/env python3
"""Streamlit UI for relation-triple inference + KG visualisation."""

from __future__ import annotations

import html
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from infer_relation_triple import _INFER_CTX, initialize_inference, predict_relation, predict_triple

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None  # type: ignore[assignment]

try:
    from pyvis.network import Network
except ModuleNotFoundError:
    Network = None  # type: ignore[assignment]

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def list_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("lambeq_relation_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def split_sentences(text: str) -> list[str]:
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        chunks.extend(part.strip() for part in SENT_SPLIT_RE.split(line) if part.strip())
    return chunks


@st.cache_resource(show_spinner=False)
def init_model(run_dir: str, spacy_model: str, normalize_input: bool) -> bool:
    initialize_inference(run_dir, spacy_model=spacy_model)
    _INFER_CTX["normalize_input"] = bool(normalize_input)
    return True


def infer_sentences(sentences: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    triples: list[dict[str, str]] = []

    for sentence in sentences:
        try:
            relation = predict_relation(sentence)
            head, pred_rel, tail = predict_triple(sentence)
            rows.append(
                {
                    "sentence": sentence,
                    "head": head,
                    "relation": pred_rel,
                    "tail": tail,
                    "confidence": relation["confidence"],
                    "status": "ok",
                }
            )
            triples.append({"head": head, "relation": pred_rel, "tail": tail})
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "sentence": sentence,
                    "head": "",
                    "relation": "",
                    "tail": "",
                    "confidence": 0.0,
                    "status": f"error: {exc}",
                }
            )

    return rows, triples


def build_kg_html(triples: list[dict[str, str]]) -> str:
    if not triples:
        return "<p>No triples to visualize.</p>"

    if Network is None or nx is None:
        items = "".join(
            f"<li>({html.escape(t['head'])}) -[{html.escape(t['relation'])}]-> ({html.escape(t['tail'])})</li>"
            for t in triples
        )
        return (
            "<p><b>pyvis/networkx not installed.</b> Showing text graph instead.</p>"
            f"<ul>{items}</ul>"
        )

    graph = nx.DiGraph()
    for triple in triples:
        head = triple["head"].strip()
        relation = triple["relation"].strip()
        tail = triple["tail"].strip()
        if not head or not relation or not tail:
            continue

        if graph.has_edge(head, tail):
            existing = graph[head][tail].get("label", "")
            labels = {label.strip() for label in existing.split("/") if label.strip()}
            labels.add(relation)
            graph[head][tail]["label"] = " / ".join(sorted(labels))
        else:
            graph.add_edge(head, tail, label=relation)

    net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="#1f2937")
    net.from_nx(graph)
    net.repulsion(node_distance=220, spring_length=160, damping=0.92)

    for node in net.nodes:
        node["shape"] = "dot"
        node["size"] = 18
        node["color"] = "#2563eb"
        node["font"] = {"size": 18, "color": "#111827"}

    for edge in net.edges:
        edge["arrows"] = "to"
        edge["color"] = "#334155"
        edge["font"] = {"align": "middle", "size": 14}

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
        net.write_html(f.name)
        temp_path = Path(f.name)

    html_text = temp_path.read_text(encoding="utf-8")
    temp_path.unlink(missing_ok=True)
    return html_text


def main() -> None:
    st.set_page_config(page_title="Relation Triple + KG", layout="wide")

    st.title("Relation Triple Inference + Knowledge Graph")
    st.caption("Input one sentence or a full paragraph, extract triples, and visualize the KG.")

    project_root = Path(__file__).resolve().parent
    runs_root = project_root / "runs"
    run_dirs = list_run_dirs(runs_root)

    with st.sidebar:
        st.header("Model")
        if run_dirs:
            run_dir = st.selectbox("Run directory", options=[str(p) for p in run_dirs], index=0)
        else:
            run_dir = st.text_input("Run directory", value=str(runs_root / "lambeq_relation_YYYYMMDD_HHMMSS"))

        spacy_model = st.text_input("spaCy model", value="en_core_web_sm")
        normalize_input = st.checkbox("Normalize input text", value=False)

        if st.button("Initialize model", type="primary"):
            try:
                init_model(run_dir, spacy_model, normalize_input)
                st.success(f"Model initialized from: {run_dir}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Initialization failed: {exc}")

    if not _INFER_CTX:
        st.info("Initialize a model from the sidebar first.")
        return

    st.subheader("Input")
    text = st.text_area(
        "Sentence or paragraph",
        height=180,
        placeholder="Example: Olivia actively trains at Yale daily. Xavier regularly operates Tableau effectively.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        split_mode = st.radio("Input mode", ["Auto sentence split", "Treat as single sentence"], horizontal=True)
    with col2:
        show_raw = st.checkbox("Show raw JSON predictions", value=False)

    if st.button("Extract triples", type="primary"):
        if not text.strip():
            st.warning("Please enter input text.")
            return

        if split_mode == "Treat as single sentence":
            sentences = [text.strip()]
        else:
            sentences = split_sentences(text)

        if not sentences:
            st.warning("No valid sentence found.")
            return

        with st.spinner("Running inference..."):
            rows, triples = infer_sentences(sentences)

        df = pd.DataFrame(rows)
        st.subheader("Predicted Triples")
        st.dataframe(df, use_container_width=True)

        ok_rows = [r for r in rows if r["status"] == "ok"]
        error_rows = [r for r in rows if r["status"] != "ok"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Sentences", len(rows))
        c2.metric("Triples extracted", len(ok_rows))
        c3.metric("Errors", len(error_rows))

        st.subheader("Knowledge Graph")
        kg_html = build_kg_html(triples)
        st.components.v1.html(kg_html, height=620, scrolling=True)

        if show_raw:
            st.subheader("Raw Results")
            st.code(json.dumps(rows, indent=2), language="json")


if __name__ == "__main__":
    main()
