#!/usr/bin/env python3
"""Single-sentence relation + triple inference for the trained lambeq RE model."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

try:
    import spacy
except ModuleNotFoundError as exc:
    raise SystemExit(
        "spaCy is required for entity extraction. Install with: pip install spacy"
    ) from exc

from lambeq import BobcatParser, RemoveCupsRewriter

# Reuse the exact training-time model and preprocessing helpers.
from train_lambeq_relation_classifier import (RelationClassifier, Sample,
                                            attach_missing_entities_from_csv,
                                            build_circuits, keep_supported_relations,
                                            load_samples_from_pickle,
                                            simplify_diagrams, stratified_split,
                                            subset)


LOGGER = logging.getLogger("re_inference")
_INFER_CTX: dict[str, Any] = {}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict relation and triple for a single sentence."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run folder containing best_model.pt and metrics_summary.json.",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        required=True,
        help="Input sentence for inference.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model with dependency parser.",
    )
    parser.add_argument(
        "--normalize-input",
        action="store_true",
        help="Lowercase + strip punctuation for model input text.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)


def _dict_to_namespace(cfg: dict[str, Any]) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in cfg.items():
        setattr(ns, k, v)
    return ns


def _load_spacy_nlp(model_name: str):
    try:
        nlp = spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load spaCy model '{model_name}'. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exc

    if "parser" not in nlp.pipe_names:
        raise RuntimeError(
            f"spaCy model '{model_name}' has no dependency parser component."
        )
    return nlp


def _rebuild_model_from_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "metrics_summary.json"
    ckpt_path = run_dir / "best_model.pt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing metrics summary: {summary_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("config", {})
    if not config:
        raise ValueError("metrics_summary.json missing 'config'.")

    relation_order = summary.get("relations")
    if not relation_order:
        raise ValueError("metrics_summary.json missing 'relations'.")

    # Convert config to training-like namespace.
    args = _dict_to_namespace(config)
    args.dataset_csv = Path(str(config["dataset_csv"]))
    args.diagrams_pkl = Path(str(config["diagrams_pkl"]))

    dummy_logger = logging.getLogger("re_inference.model_rebuild")
    dummy_logger.setLevel(logging.WARNING)

    samples = load_samples_from_pickle(args.diagrams_pkl, dummy_logger)
    attach_missing_entities_from_csv(samples, args.dataset_csv, dummy_logger)
    samples, _ = keep_supported_relations(samples, relation_order)

    if getattr(args, "max_samples", 0) > 0 and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
    simplify_diagrams(
        samples,
        dummy_logger,
        max_error_logs=getattr(args, "rewrite_log_max_errors", 5),
    )
    samples, circuits = build_circuits(samples, args, dummy_logger)
    if not circuits:
        raise RuntimeError("No circuits available after rebuilding training pipeline.")

    train_idx, val_idx, test_idx = stratified_split(
        samples=samples,
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    train_circuits = subset(circuits, train_idx)
    val_circuits = subset(circuits, val_idx)
    test_circuits = subset(circuits, test_idx)

    backend_config: dict[str, Any] = {"backend": args.pennylane_backend}
    if int(getattr(args, "shots", 0)) > 0:
        backend_config["shots"] = int(args.shots)

    if getattr(args, "model_init_split", "train") == "all":
        init_circuits = train_circuits + val_circuits + test_circuits
    else:
        init_circuits = train_circuits
    if not init_circuits:
        raise RuntimeError("No init circuits available to reconstruct model.")

    model = RelationClassifier.from_diagrams(
        init_circuits,
        n_classes=len(relation_order),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        probabilities=True,
        normalize=True,
        backend_config=backend_config,
    )
    model.initialise_weights()

    state = torch.load(ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        # LazyLinear fallback if required.
        _ = model(init_circuits[:1])
        model.load_state_dict(state, strict=True)
    model.eval()

    parser = BobcatParser(verbose="suppress")
    rewriter = RemoveCupsRewriter()
    nlp = _load_spacy_nlp("en_core_web_sm")

    return {
        "model": model,
        "parser": parser,
        "rewriter": rewriter,
        "args": args,
        "relation_order": relation_order,
        "spacy_nlp": nlp,
        "run_dir": run_dir,
        "checkpoint_path": ckpt_path,
        "backend_config": backend_config,
        "init_circuits": init_circuits,
        "normalize_input": False,
    }


def initialize_inference(run_dir: str | Path, spacy_model: str = "en_core_web_sm") -> None:
    """Initialize global inference context from training artifacts."""
    run_path = Path(run_dir)
    ctx = _rebuild_model_from_run(run_path)
    if spacy_model != "en_core_web_sm":
        ctx["spacy_nlp"] = _load_spacy_nlp(spacy_model)
    _INFER_CTX.clear()
    _INFER_CTX.update(ctx)
    LOGGER.info("Inference initialized from run: %s", run_path)


def _require_ctx() -> dict[str, Any]:
    if not _INFER_CTX:
        raise RuntimeError("Inference is not initialized. Call initialize_inference(...) first.")
    return _INFER_CTX


def _normalize_text_for_model(sentence: str) -> str:
    return " ".join(TOKEN_RE.findall(sentence.lower()))


def _expand_model_with_new_circuits(new_circuits: list[Any]) -> None:
    """Rebuild model with additional circuits so PennyLaneModel can evaluate them."""
    ctx = _require_ctx()
    args = ctx["args"]
    relation_order = ctx["relation_order"]
    backend_config = ctx["backend_config"]
    checkpoint_path: Path = ctx["checkpoint_path"]
    base_circuits: list[Any] = ctx["init_circuits"]

    expanded = base_circuits + new_circuits
    model = RelationClassifier.from_diagrams(
        expanded,
        n_classes=len(relation_order),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        probabilities=True,
        normalize=True,
        backend_config=backend_config,
    )
    model.initialise_weights()

    state = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        LOGGER.warning(
            "Expanded model has %d missing keys (likely unseen token parameters).",
            len(incompatible.missing_keys),
        )
    model.eval()

    ctx["model"] = model
    ctx["init_circuits"] = expanded
    LOGGER.info("Expanded inference model circuit map to %d circuits.", len(expanded))


def extract_entities(sentence: str) -> tuple[str | None, str | None]:
    """Extract (head_entity, tail_entity) using spaCy dependency parsing."""
    ctx = _require_ctx()
    nlp = ctx["spacy_nlp"]
    doc = nlp(sentence)

    ents = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
    if len(ents) >= 2:
        return ents[0], ents[1]

    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
    if len(noun_chunks) >= 2:
        return noun_chunks[0], noun_chunks[-1]

    head = None
    tail = None
    for tok in doc:
        if head is None and tok.dep_ in {"nsubj", "nsubjpass", "csubj"}:
            head = tok.subtree.text.strip()
        if tail is None and tok.dep_ in {"dobj", "pobj", "obj", "attr", "dative", "oprd"}:
            tail = tok.subtree.text.strip()
    return head, tail


def predict_relation(sentence: str) -> dict[str, Any]:
    """Predict relation label and probabilities for a single sentence."""
    ctx = _require_ctx()
    model = ctx["model"]
    parser = ctx["parser"]
    rewriter = ctx["rewriter"]
    args = ctx["args"]
    relation_order = ctx["relation_order"]

    raw_text = sentence.strip()
    if ctx.get("normalize_input", False):
        text = _normalize_text_for_model(raw_text)
    else:
        text = raw_text
    if not text:
        raise ValueError("Input sentence is empty.")

    try:
        diagram = parser.sentence2diagram(text)
    except Exception as exc:
        raise RuntimeError(f"Bobcat failed to parse sentence: {text}") from exc

    try:
        diagram = rewriter(diagram)
    except Exception:
        # Non-fatal; keep original diagram if rewrite fails.
        pass

    tmp_sample = Sample(sentence=text, relation=relation_order[0], diagram=diagram)
    kept, circuits = build_circuits([tmp_sample], args, LOGGER)
    if not kept or not circuits:
        raise RuntimeError(
            "Could not build a valid circuit for this sentence "
            "(possibly filtered by max-circuit-width)."
        )

    with torch.no_grad():
        try:
            logits = model(circuits)
        except KeyError:
            LOGGER.warning(
                "Unseen circuit at inference; expanding model circuit map and retrying."
            )
            _expand_model_with_new_circuits(circuits)
            model = _require_ctx()["model"]
            logits = model(circuits)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    pred_relation = relation_order[pred_idx]

    prob_map = {relation_order[i]: float(probs[i]) for i in range(len(relation_order))}
    return {
        "sentence": text,
        "predicted_relation": pred_relation,
        "confidence": float(probs[pred_idx]),
        "probabilities": prob_map,
    }


def predict_triple(sentence: str) -> tuple[str, str, str]:
    """Return (head_entity, predicted_relation, tail_entity) for one sentence."""
    rel = predict_relation(sentence)
    head, tail = extract_entities(sentence)

    if not head or not tail:
        raise ValueError(
            "Could not detect both head and tail entities from sentence. "
            "Try a clearer sentence with explicit entity mentions."
        )
    return head, rel["predicted_relation"], tail


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    initialize_inference(args.run_dir, args.spacy_model)
    _INFER_CTX["normalize_input"] = bool(args.normalize_input)

    try:
        rel = predict_relation(args.sentence)
        head, tail = extract_entities(args.sentence)
        print("Relation Prediction:")
        print(json.dumps(rel, indent=2))

        if head and tail:
            triple = (head, rel["predicted_relation"], tail)
            print("\nPredicted Triple:")
            print(triple)
        else:
            print(
                "\nPredicted Triple:\n"
                "Entity detection failed: could not detect both head and tail."
            )
    except Exception as exc:
        LOGGER.exception("Inference failed.")
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

# Example CLI usage:
# python infer_relation_triple.py \
#   --run-dir runs/lambeq_relation_YYYYMMDD_HHMMSS \
#   --sentence "Alice works at Google in London."

# Example Python usage:
# from infer_relation_triple import initialize_inference, predict_relation, extract_entities, predict_triple
# initialize_inference("runs/lambeq_relation_YYYYMMDD_HHMMSS")
# print(predict_relation("Alice works at Google in London."))
# print(extract_entities("Alice works at Google in London."))
# print(predict_triple("Alice works at Google in London."))