#!/usr/bin/env python3
"""DisCoCat QNLP relation classification pipeline (lambeq + PennyLane)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch import nn

try:
    from lambeq import (AtomicType, BobcatParser, Dataset, IQPAnsatz,
                        PennyLaneModel, RemoveCupsRewriter)
except ModuleNotFoundError as exc:
    raise SystemExit(
        "lambeq is required to run this pipeline. Install dependencies first "
        "(see requirements-lambeq.txt)."
    ) from exc


@dataclass
class Sample:
    sentence: str
    relation: str
    diagram: Any
    head: str | None = None
    tail: str | None = None


class RelationClassifier(PennyLaneModel):
    """Hybrid model: quantum circuit outputs -> small classifier head."""

    def __init__(
        self,
        n_classes: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.classifier = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, diagrams):  # type: ignore[override]
        probs = self.get_diagram_output(diagrams)
        # PennyLaneModel can return higher-rank tensors (e.g. [B, 2, 2, 2]).
        # Classifier/CrossEntropy expects [B, F] -> [B, C].
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        elif probs.ndim > 2:
            probs = probs.reshape(probs.shape[0], -1)
        # Center [0, 1] probability features for better optimization.
        features = 2 * (probs - 0.5)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lambeq+PennyLane relation classifier."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("data/relation_extraction_discocat.csv"),
        help="CSV with sentence, relation, entity_1, entity_2.",
    )
    parser.add_argument(
        "--diagrams-pkl",
        type=Path,
        default=Path("semeval_bobcat_diagrams.pkl"),
        help="Pickle with Bobcat output. If missing, parser is run on CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Root folder for logs, checkpoints, and predictions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for console and file logs.",
    )
    parser.add_argument(
        "--parse-log-every",
        type=int,
        default=500,
        help="Log CSV parse progress every N sentences (0 disables).",
    )
    parser.add_argument(
        "--circuit-log-every",
        type=int,
        default=500,
        help="Log circuit-build progress every N samples (0 disables).",
    )
    parser.add_argument(
        "--rewrite-log-max-errors",
        type=int,
        default=5,
        help="Maximum diagram rewrite failures to log with traceback context.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--train-log-every-batches",
        type=int,
        default=0,
        help="Log training batch progress every N batches (0 disables).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, keep only first N samples after relation filtering.",
    )
    parser.add_argument(
        "--max-circuit-width",
        type=int,
        default=0,
        help="If >0, drop samples whose generated circuit width exceeds this value.",
    )
    parser.add_argument(
        "--model-init-split",
        choices=["train", "all"],
        default="train",
        help="Circuits used for RelationClassifier.from_diagrams (train saves memory).",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-single-qubit-params", type=int, default=3)
    parser.add_argument("--n-qubits-noun", type=int, default=1)
    parser.add_argument("--n-qubits-sentence", type=int, default=3)
    parser.add_argument(
        "--n-qubits-preposition",
        type=int,
        default=1,
        help="Qubits used for preposition type (AtomicType.PREPOSITION / Ty(p)).",
    )
    parser.add_argument(
        "--n-qubits-other",
        type=int,
        default=1,
        help="Default qubits for unseen atomic types (e.g., Ty(conj), Ty(det)).",
    )
    parser.add_argument(
        "--pennylane-backend",
        default="default.qubit",
        help="PennyLane backend, e.g. default.qubit.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Set >0 for shot-based simulation. 0 means analytic mode.",
    )
    parser.add_argument(
        "--kg-confidence-threshold",
        type=float,
        default=0.50,
        help="Min confidence to keep predicted triples for KG output.",
    )
    parser.add_argument(
        "--epoch-sample-count",
        type=int,
        default=3,
        help="How many validation examples to log after each epoch.",
    )
    parser.add_argument(
        "--relation-order",
        nargs="+",
        default=None,
        help=(
            "Explicit class order for label mapping. "
            "If omitted, relations are inferred from loaded samples."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"lambeq_relation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("lambeq_relation")
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "training.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def read_csv_rows(dataset_csv: Path) -> list[dict[str, str]]:
    with dataset_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def log_relation_distribution(
    samples: list[Sample], logger: logging.Logger, title: str
) -> None:
    counts = Counter(s.relation for s in samples)
    logger.info("%s | classes=%d | distribution=%s", title, len(counts), dict(counts))


def load_samples_from_pickle(diagrams_pkl: Path, logger: logging.Logger) -> list[Sample]:
    with diagrams_pkl.open("rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected pickle to contain a list of parsed samples.")

    samples: list[Sample] = []
    skipped = 0
    for item in raw:
        if not isinstance(item, dict):
            skipped += 1
            continue

        diagram = item.get("diagram")
        relation = item.get("relation")
        sentence = item.get("sentence") or item.get("simplified_sentence")
        if diagram is None or relation is None or sentence is None:
            skipped += 1
            continue

        samples.append(
            Sample(
                sentence=str(sentence),
                relation=str(relation),
                diagram=diagram,
                head=_opt_str(item.get("head"))
                or _opt_str(item.get("head_entity"))
                or _opt_str(item.get("entity_1")),
                tail=_opt_str(item.get("tail"))
                or _opt_str(item.get("tail_entity"))
                or _opt_str(item.get("entity_2")),
            )
        )

    logger.info(
        "Loaded diagram pickle entries: total=%d valid=%d skipped=%d",
        len(raw),
        len(samples),
        skipped,
    )
    log_relation_distribution(samples, logger, "Loaded sample relations")
    return samples


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def parse_samples_from_csv(
    dataset_csv: Path, logger: logging.Logger, parse_log_every: int = 0
) -> list[Sample]:
    rows = read_csv_rows(dataset_csv)
    parser = BobcatParser(verbose="text")

    samples: list[Sample] = []
    failures = 0
    for i, row in enumerate(rows, start=1):
        sentence = row["sentence"]
        try:
            diagram = parser.sentence2diagram(sentence)
            samples.append(
                Sample(
                    sentence=sentence,
                    relation=row["relation"],
                    head=_opt_str(row.get("entity_1")),
                    tail=_opt_str(row.get("entity_2")),
                    diagram=diagram,
                )
            )
        except Exception:
            failures += 1
            if failures <= 5:
                logger.exception("Bobcat parsing failed for sentence: %s", sentence)
        if parse_log_every > 0 and i % parse_log_every == 0:
            logger.info(
                "CSV parse progress: %d/%d | parsed=%d | failures=%d",
                i,
                len(rows),
                len(samples),
                failures,
            )

    logger.info("Bobcat parsing failures: %d", failures)
    log_relation_distribution(samples, logger, "Parsed sample relations")
    return samples


def attach_missing_entities_from_csv(
    samples: list[Sample], dataset_csv: Path, logger: logging.Logger
) -> None:
    rows = read_csv_rows(dataset_csv)
    by_pair = {
        (row["sentence"], row["relation"]): (row.get("entity_1"), row.get("entity_2"))
        for row in rows
    }

    filled = 0
    for sample in samples:
        if sample.head and sample.tail:
            continue
        ent = by_pair.get((sample.sentence, sample.relation))
        if ent:
            sample.head = sample.head or _opt_str(ent[0])
            sample.tail = sample.tail or _opt_str(ent[1])
            filled += 1

    logger.info("Entity metadata filled from CSV: %d", filled)


def keep_supported_relations(
    samples: list[Sample], relation_order: list[str] | None
) -> tuple[list[Sample], list[str]]:
    if not relation_order:
        # Keep encounter order for deterministic mapping.
        seen: set[str] = set()
        inferred: list[str] = []
        for sample in samples:
            if sample.relation not in seen:
                seen.add(sample.relation)
                inferred.append(sample.relation)
        return samples, inferred

    allowed = set(relation_order)
    filtered = [s for s in samples if s.relation in allowed]
    present = {s.relation for s in filtered}
    ordered_present = [r for r in relation_order if r in present]
    return filtered, ordered_present


def simplify_diagrams(
    samples: list[Sample], logger: logging.Logger, max_error_logs: int = 5
) -> None:
    rewriter = RemoveCupsRewriter()
    changed = 0
    failed = 0
    for i, sample in enumerate(samples, start=1):
        try:
            sample.diagram = rewriter(sample.diagram)
            changed += 1
        except Exception:
            failed += 1
            if failed <= max_error_logs:
                logger.exception(
                    "Diagram rewrite failed at sample %d | relation=%s | sentence=%s",
                    i,
                    sample.relation,
                    sample.sentence,
                )
    logger.info(
        "Diagrams simplified with RemoveCupsRewriter: success=%d failed=%d",
        changed,
        failed,
    )


def build_circuits(
    samples: list[Sample], args: argparse.Namespace, logger: logging.Logger
) -> tuple[list[Sample], list[Any]]:
    wire_map = {
        AtomicType.NOUN: args.n_qubits_noun,
        AtomicType.SENTENCE: args.n_qubits_sentence,
    }
    # Bobcat diagrams may include Ty(p). Map it when available.
    if hasattr(AtomicType, "PREPOSITION"):
        wire_map[getattr(AtomicType, "PREPOSITION")] = args.n_qubits_preposition
    else:
        # Backward-compatibility for versions without AtomicType.PREPOSITION.
        try:
            preposition_ty = type(AtomicType.NOUN)("p")
            wire_map[preposition_ty] = args.n_qubits_preposition
        except Exception:
            pass

    def _make_ansatz() -> IQPAnsatz:
        return IQPAnsatz(
            wire_map,
            n_layers=args.n_layers,
            n_single_qubit_params=args.n_single_qubit_params,
        )

    ansatz = _make_ansatz()
    circuits: list[Any] = []
    kept_samples: list[Sample] = []
    dropped_for_width = 0
    dropped_for_errors = 0

    for i, sample in enumerate(samples, start=1):
        while True:
            try:
                circuit = ansatz(sample.diagram)
                width = get_circuit_width(circuit)
                if args.max_circuit_width > 0 and width > args.max_circuit_width:
                    dropped_for_width += 1
                    break
                circuits.append(circuit)
                kept_samples.append(sample)
                if args.circuit_log_every > 0 and i % args.circuit_log_every == 0:
                    logger.info(
                        "Circuit build progress: %d/%d | kept=%d dropped_width=%d "
                        "| dynamic_types=%d",
                        i,
                        len(samples),
                        len(circuits),
                        dropped_for_width,
                        max(0, len(wire_map) - 3),
                    )
                break
            except KeyError as exc:
                if not exc.args:
                    raise
                missing_ty = exc.args[0]
                if missing_ty in wire_map:
                    raise
                wire_map[missing_ty] = args.n_qubits_other
                logger.info(
                    "Added dynamic wire_map entry: %s -> %d qubits",
                    missing_ty,
                    args.n_qubits_other,
                )
                ansatz = _make_ansatz()
            except Exception:
                logger.exception(
                    "Circuit build failed at sample %d | relation=%s | sentence=%s",
                    i,
                    sample.relation,
                    sample.sentence,
                )
                dropped_for_errors += 1
                break

    if circuits:
        widths = np.array([get_circuit_width(c) for c in circuits], dtype=int)
        logger.info(
            "Circuit width stats | min=%d p50=%d p90=%d p95=%d max=%d",
            int(widths.min()),
            int(np.percentile(widths, 50)),
            int(np.percentile(widths, 90)),
            int(np.percentile(widths, 95)),
            int(widths.max()),
        )
    logger.info(
        "Circuit build summary | input=%d kept=%d dropped_width=%d dropped_errors=%d",
        len(samples),
        len(circuits),
        dropped_for_width,
        dropped_for_errors,
    )

    return kept_samples, circuits


def get_circuit_width(circuit: Any) -> int:
    """Best-effort circuit width extraction across lambeq versions."""
    for attr in ("cod", "dom"):
        if hasattr(circuit, attr):
            ty = getattr(circuit, attr)
            try:
                return int(len(ty))
            except Exception:
                pass
    if hasattr(circuit, "n_qubits"):
        try:
            return int(getattr(circuit, "n_qubits"))
        except Exception:
            pass
    return 0


def stratified_split(
    samples: list[Sample], val_size: float, test_size: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("Require 0 < val_size, test_size and val_size+test_size < 1.")

    y = np.array([s.relation for s in samples])
    idx = np.arange(len(samples))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=val_size + test_size,
        stratify=y,
        random_state=seed,
    )
    temp_y = y[temp_idx]
    val_ratio = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_ratio,
        stratify=temp_y,
        random_state=seed,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def build_label_maps(relations: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    r2i = {rel: i for i, rel in enumerate(relations)}
    i2r = {i: rel for rel, i in r2i.items()}
    return r2i, i2r


def subset(items: list[Any], indices: list[int]) -> list[Any]:
    return [items[i] for i in indices]


def compute_metrics(
    y_true: list[int], y_pred: list[int], n_classes: int
) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "num_classes": float(n_classes),
    }


def evaluate_split(
    model: RelationClassifier,
    circuits: list[Any],
    labels: list[int],
    n_classes: int,
) -> tuple[float, dict[str, float], list[int], np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(circuits)
        y = torch.tensor(labels, dtype=torch.long)
        loss = F.cross_entropy(logits, y).item()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1).tolist()

    metrics = compute_metrics(labels, preds, n_classes=n_classes)
    return loss, metrics, preds, probs


def train(
    model: RelationClassifier,
    train_dataset: Dataset,
    train_circuits: list[Any],
    train_labels: list[int],
    val_samples: list[Sample],
    val_circuits: list[Any],
    val_labels: list[int],
    id2relation: dict[int, str],
    args: argparse.Namespace,
    logger: logging.Logger,
    run_dir: Path,
    n_classes: int,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    # Initialize lazy layers.
    _ = model(train_circuits[:1])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_epoch = -1
    best_state_path = run_dir / "best_model.pt"
    patience_ctr = 0
    num_train_batches = max(1, int(np.ceil(len(train_labels) / args.batch_size)))

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        y_true_epoch: list[int] = []
        y_pred_epoch: list[int] = []

        for batch_i, (circuits_batch, labels_batch) in enumerate(train_dataset, start=1):
            try:
                y = torch.tensor(labels_batch, dtype=torch.long)
                optimizer.zero_grad()
                logits = model(circuits_batch)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
            except Exception:
                logger.exception(
                    "Training failed at epoch=%d batch=%d/%d batch_size=%d",
                    epoch,
                    batch_i,
                    num_train_batches,
                    len(labels_batch),
                )
                raise

            total_loss += loss.item() * len(labels_batch)
            preds = torch.argmax(logits.detach(), dim=1).cpu().tolist()
            y_true_epoch.extend(y.cpu().tolist())
            y_pred_epoch.extend(preds)
            if (
                args.train_log_every_batches > 0
                and batch_i % args.train_log_every_batches == 0
            ):
                logger.info(
                    "Epoch %d batch progress: %d/%d",
                    epoch,
                    batch_i,
                    num_train_batches,
                )

        train_loss = total_loss / max(1, len(train_labels))
        train_metrics = compute_metrics(y_true_epoch, y_pred_epoch, n_classes)

        val_loss, val_metrics, _, _ = evaluate_split(
            model=model,
            circuits=val_circuits,
            labels=val_labels,
            n_classes=n_classes,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "train_precision_macro": train_metrics["precision_macro"],
            "val_precision_macro": val_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
        }
        history.append(row)

        logger.info(
            "Epoch %d | train_loss=%.4f val_loss=%.4f | train_acc=%.4f val_acc=%.4f "
            "| train_f1=%.4f val_f1=%.4f",
            epoch,
            train_loss,
            val_loss,
            train_metrics["accuracy"],
            val_metrics["accuracy"],
            train_metrics["f1_macro"],
            val_metrics["f1_macro"],
        )
        logger.info("Epoch %d completed in %.2fs", epoch, time.perf_counter() - epoch_start)
        log_epoch_samples(
            model=model,
            epoch=epoch,
            val_samples=val_samples,
            val_circuits=val_circuits,
            val_labels=val_labels,
            id2relation=id2relation,
            sample_count=args.epoch_sample_count,
            logger=logger,
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), best_state_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (best epoch=%d, best val_f1=%.4f).",
                    epoch,
                    best_epoch,
                    best_f1,
                )
                break

    if best_state_path.exists():
        model.load_state_dict(torch.load(best_state_path, map_location="cpu"))
        logger.info("Loaded best checkpoint from epoch %d.", best_epoch)

    return history


def log_epoch_samples(
    model: RelationClassifier,
    epoch: int,
    val_samples: list[Sample],
    val_circuits: list[Any],
    val_labels: list[int],
    id2relation: dict[int, str],
    sample_count: int,
    logger: logging.Logger,
) -> None:
    if sample_count <= 0 or not val_samples:
        return

    k = min(sample_count, len(val_samples))
    idxs = list(range(k))
    sample_objs = [val_samples[i] for i in idxs]
    sample_circuits = [val_circuits[i] for i in idxs]
    sample_labels = [val_labels[i] for i in idxs]

    model.eval()
    with torch.no_grad():
        logits = model(sample_circuits)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1).tolist()

    logger.info("Epoch %d | sample input/output:", epoch)
    for i, (sample, true_id, pred_id, p) in enumerate(
        zip(sample_objs, sample_labels, preds, probs), start=1
    ):
        logger.info(
            "[%d] input=\"%s\" | true=%s | pred=%s | conf=%.4f",
            i,
            sample.sentence,
            id2relation[true_id],
            id2relation[pred_id],
            float(np.max(p)),
        )


def write_history_csv(history: list[dict[str, float]], out_file: Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_confusion_matrix(
    cm: np.ndarray, id2relation: dict[int, str], out_file: Path
) -> None:
    labels = [id2relation[i] for i in range(len(id2relation))]
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm.tolist()):
            writer.writerow([labels[i]] + row)


def write_predictions_csv(
    samples: list[Sample],
    y_true: list[int],
    y_pred: list[int],
    probs: np.ndarray,
    id2relation: dict[int, str],
    out_file: Path,
) -> None:
    rel_order = [id2relation[i] for i in range(len(id2relation))]
    fieldnames = [
        "sentence",
        "head",
        "tail",
        "true_relation",
        "pred_relation",
        "confidence",
    ] + [f"prob_{rel}" for rel in rel_order]

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample, true_id, pred_id, p in zip(samples, y_true, y_pred, probs):
            row = {
                "sentence": sample.sentence,
                "head": sample.head or "",
                "tail": sample.tail or "",
                "true_relation": id2relation[true_id],
                "pred_relation": id2relation[pred_id],
                "confidence": float(np.max(p)),
            }
            for rel, prob in zip(rel_order, p.tolist()):
                row[f"prob_{rel}"] = float(prob)
            writer.writerow(row)


def write_kg_triples_jsonl(
    samples: list[Sample],
    y_pred: list[int],
    probs: np.ndarray,
    id2relation: dict[int, str],
    threshold: float,
    out_file: Path,
) -> None:
    with out_file.open("w", encoding="utf-8") as f:
        for sample, pred_id, p in zip(samples, y_pred, probs):
            conf = float(np.max(p))
            if conf < threshold:
                continue
            if not sample.head or not sample.tail:
                continue
            obj = {
                "head": sample.head,
                "relation": id2relation[pred_id],
                "tail": sample.tail,
                "confidence": conf,
                "sentence": sample.sentence,
            }
            f.write(json.dumps(obj) + "\n")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = setup_run_dir(args.output_root)
    logger = setup_logging(run_dir, args.log_level)
    logger.info("Run directory: %s", run_dir)
    logger.info("Arguments: %s", vars(args))
    try:
        t0 = time.perf_counter()
        logger.info("Stage: load_samples")
        if args.diagrams_pkl.exists():
            logger.info("Loading pre-parsed diagrams from: %s", args.diagrams_pkl)
            samples = load_samples_from_pickle(args.diagrams_pkl, logger)
        else:
            logger.info(
                "No pickle found at %s; parsing sentences from CSV with Bobcat.",
                args.diagrams_pkl,
            )
            samples = parse_samples_from_csv(
                args.dataset_csv, logger, parse_log_every=args.parse_log_every
            )
        logger.info(
            "Stage complete: load_samples | elapsed=%.2fs",
            time.perf_counter() - t0,
        )

        if not samples:
            raise SystemExit("No valid samples available for training.")

        t0 = time.perf_counter()
        logger.info("Stage: preprocess_samples")
        attach_missing_entities_from_csv(samples, args.dataset_csv, logger)
        available_relations = sorted({s.relation for s in samples})
        samples, relation_order = keep_supported_relations(samples, args.relation_order)
        if len(relation_order) < 2:
            raise SystemExit(
                "Need at least two relations after filtering. "
                f"Available relations in loaded diagrams: {available_relations}. "
                "Pass --relation-order with valid labels or omit it to auto-infer."
            )
        logger.info("Using relations: %s", relation_order)
        logger.info("Total samples after filtering: %d", len(samples))
        if args.max_samples > 0 and len(samples) > args.max_samples:
            samples = samples[: args.max_samples]
            logger.warning(
                "Applied max-samples=%d, reduced dataset to %d samples.",
                args.max_samples,
                len(samples),
            )
            log_relation_distribution(samples, logger, "Post max-samples relations")
        simplify_diagrams(samples, logger, max_error_logs=args.rewrite_log_max_errors)
        logger.info(
            "Stage complete: preprocess_samples | elapsed=%.2fs",
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        logger.info("Stage: build_circuits")
        samples, circuits = build_circuits(samples, args, logger)
        if len(samples) < 2:
            raise SystemExit(
                "Too few samples after circuit filtering. "
                "Increase --max-circuit-width or disable it with 0."
            )
        log_relation_distribution(samples, logger, "Post circuit-filter relations")
        logger.info(
            "Stage complete: build_circuits | elapsed=%.2fs | circuits=%d",
            time.perf_counter() - t0,
            len(circuits),
        )

        t0 = time.perf_counter()
        logger.info("Stage: split_dataset")
        try:
            train_idx, val_idx, test_idx = stratified_split(
                samples=samples,
                val_size=args.val_size,
                test_size=args.test_size,
                seed=args.seed,
            )
        except ValueError as exc:
            raise SystemExit(
                "Failed stratified split after filtering. "
                "Use more samples or relax max-circuit-width."
            ) from exc

        relation2id, id2relation = build_label_maps(relation_order)
        labels = [relation2id[s.relation] for s in samples]

        train_samples, val_samples, test_samples = (
            subset(samples, train_idx),
            subset(samples, val_idx),
            subset(samples, test_idx),
        )
        train_circuits, val_circuits, test_circuits = (
            subset(circuits, train_idx),
            subset(circuits, val_idx),
            subset(circuits, test_idx),
        )
        train_labels, val_labels, test_labels = (
            subset(labels, train_idx),
            subset(labels, val_idx),
            subset(labels, test_idx),
        )

        logger.info(
            "Split sizes | train=%d val=%d test=%d",
            len(train_labels),
            len(val_labels),
            len(test_labels),
        )
        logger.info(
            "Stage complete: split_dataset | elapsed=%.2fs",
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        logger.info("Stage: init_model")
        backend_config: dict[str, Any] = {"backend": args.pennylane_backend}
        if args.shots > 0:
            backend_config["shots"] = args.shots

        if args.model_init_split == "all":
            init_circuits = train_circuits + val_circuits + test_circuits
        else:
            init_circuits = train_circuits
        logger.info(
            "Model init circuits source=%s count=%d",
            args.model_init_split,
            len(init_circuits),
        )
        model = RelationClassifier.from_diagrams(
            init_circuits,
            n_classes=len(relation_order),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            probabilities=True,
            normalize=True,
            backend_config=backend_config,
        )
        model.initialise_weights()

        train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=args.batch_size,
            shuffle=True,
        )
        logger.info(
            "Stage complete: init_model | elapsed=%.2fs",
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        logger.info("Stage: train")
        history = train(
            model=model,
            train_dataset=train_dataset,
            train_circuits=train_circuits,
            train_labels=train_labels,
            val_samples=val_samples,
            val_circuits=val_circuits,
            val_labels=val_labels,
            id2relation=id2relation,
            args=args,
            logger=logger,
            run_dir=run_dir,
            n_classes=len(relation_order),
        )
        write_history_csv(history, run_dir / "epoch_metrics.csv")
        logger.info("Stage complete: train | elapsed=%.2fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        logger.info("Stage: evaluate_and_export")
        train_eval = evaluate_split(
            model, train_circuits, train_labels, n_classes=len(relation_order)
        )
        val_eval = evaluate_split(
            model, val_circuits, val_labels, n_classes=len(relation_order)
        )
        test_eval = evaluate_split(
            model, test_circuits, test_labels, n_classes=len(relation_order)
        )

        _, train_metrics, _, _ = train_eval
        _, val_metrics, _, _ = val_eval
        test_loss, test_metrics, test_preds, test_probs = test_eval

        cm = confusion_matrix(
            test_labels, test_preds, labels=list(range(len(relation_order)))
        )
        report = classification_report(
            test_labels,
            test_preds,
            labels=list(range(len(relation_order))),
            target_names=[id2relation[i] for i in range(len(relation_order))],
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        summary = {
            "config": vars(args),
            "run_dir": str(run_dir),
            "relations": relation_order,
            "label_mapping": relation2id,
            "split_sizes": {
                "train": len(train_labels),
                "val": len(val_labels),
                "test": len(test_labels),
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": {"loss": test_loss, **test_metrics},
            "classification_report_test": report,
        }

        with (run_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(summary), f, indent=2)
        write_confusion_matrix(cm, id2relation, run_dir / "confusion_matrix_test.csv")
        write_predictions_csv(
            samples=test_samples,
            y_true=test_labels,
            y_pred=test_preds,
            probs=test_probs,
            id2relation=id2relation,
            out_file=run_dir / "test_predictions.csv",
        )
        write_kg_triples_jsonl(
            samples=test_samples,
            y_pred=test_preds,
            probs=test_probs,
            id2relation=id2relation,
            threshold=args.kg_confidence_threshold,
            out_file=run_dir / "kg_triples_test.jsonl",
        )

        logger.info("Final test metrics: %s", {"loss": test_loss, **test_metrics})
        logger.info("Stage complete: evaluate_and_export | elapsed=%.2fs", time.perf_counter() - t0)
        logger.info("Artifacts saved in: %s", run_dir)
    except Exception:
        logger.exception("Pipeline failed with an unhandled exception.")
        raise


if __name__ == "__main__":
    main()