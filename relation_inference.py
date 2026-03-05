#!/usr/bin/env python3
"""Inference helpers for the lambeq+PennyLane relation classifier.

This script is intended to sit alongside
``train_lambeq_relation_classifier.py`` and re‑uses most of the
preprocessing pipeline that the trainer employs.  The workflow is
roughly:

  1. parse a CSV containing the training data (or a pickle produced when
     training); this is only used to construct the circuit map, not to
     actually score the samples.
  2. simplify the diagrams and build the quantum circuits exactly as the
     trainer does, capturing the final ``wire_map`` used to handle
     dynamic atomic types.
  3. instantiate a ``RelationClassifier`` with those circuits and load a
     saved state dictionary from a checkpoint file.

Once the pipeline has been initialised the three public functions below
can be used for single‑sentence inference.

Functions
---------

* ``extract_entities(sentence)`` – run a static spaCy dependency parse
  and return a (head, tail) pair of strings.
* ``predict_relation(sentence, model, parser, rewriter, ansatz, rel_order)`` –
  convert a sentence to a diagram/circuit and run the neural+quantum
  model, returning the predicted relation (plus confidence).
* ``predict_triple(sentence, ...)`` – combine the two functions above and
  return ``(head, relation, tail)``.

The ``main`` block at the bottom shows how you might load a checkpoint
and perform a few example predictions; it also demonstrates the minimal
argument parsing required to keep the pipeline in sync with the training
script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np
import spacy
import torch
from lambeq import (
    AtomicType,
    BobcatParser,
    IQPAnsatz,
    RemoveCupsRewriter,
    Dataset,
)
from torch import nn
from torch.nn import functional as F

# import some helpers from the training module so that we don't duplicate
# behaviour; the training file lives in the same directory so make sure it
# is on the path.
sys.path.append(str(Path(__file__).parent))
from train_lambeq_relation_classifier import (
    parse_samples_from_csv,
    keep_supported_relations,
    simplify_diagrams,
    Sample,
    RelationClassifier,
)


# ---------------------------------------------------------------------------
# utility functions taken/adapted from the trainer
# ---------------------------------------------------------------------------


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


# the trainer keeps this logic in build_circuits; we adapt a subset so that we
# can capture the final wire_map and re‑use it when building new sentences

def _make_ansatz_factory(
    args: argparse.Namespace, wire_map: dict[Any, int]
) -> Callable[[], IQPAnsatz]:
    def factory() -> IQPAnsatz:
        return IQPAnsatz(
            wire_map,
            n_layers=args.n_layers,
            n_single_qubit_params=args.n_single_qubit_params,
        )

    return factory


# This routine mimics the "build_circuits" helper from the training script
# but returns extra objects needed for inference (wire_map + ansatz maker).

def prepare_pipeline(
    dataset_csv: Path,
    checkpoint: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[
    RelationClassifier,
    List[str],
    BobcatParser,
    RemoveCupsRewriter,
    Callable[[], IQPAnsatz],
]:
    """Initialise model + diagram pipeline from a dataset and checkpoint.

    Parameters
    ----------
    dataset_csv:
        CSV file used during training.  The file itself has no bearing on
        inference except that it is re‑parsed so that the circuit_map
        learned by :class:`RelationClassifier` contains every atomic type
        that the model may later encounter.
    checkpoint:
        Path to the ``.pt`` file saved by the training script.
    args:
        Namespace containing the same command‑line options that were used
        during training; see ``train_lambeq_relation_classifier.parse_args``
        for the full list.  In particular this provides qubit counts,
        backend settings and relation ordering.
    logger:
        Logger used for informational messages; the caller may configure
        handlers as desired.

    Returns
    -------
    model, relation_order, parser, rewriter, ansatz_factory
    """

    # --- mimic early stages of the trainer ----------------------------------------------------------------
    samples = parse_samples_from_csv(dataset_csv, logger, parse_log_every=0)
    samples, relation_order = keep_supported_relations(samples, args.relation_order)
    simplify_diagrams(samples, logger, max_error_logs=args.rewrite_log_max_errors)

    # build circuits and capture the final wire_map
    wire_map: dict[Any, int] = {
        AtomicType.NOUN: args.n_qubits_noun,
        AtomicType.SENTENCE: args.n_qubits_sentence,
    }
    if hasattr(AtomicType, "PREPOSITION"):
        wire_map[getattr(AtomicType, "PREPOSITION")] = args.n_qubits_preposition
    else:  # backward compatibility
        try:
            preposition_ty = type(AtomicType.NOUN)("p")
            wire_map[preposition_ty] = args.n_qubits_preposition
        except Exception:
            pass

    ansatz = _make_ansatz_factory(args, wire_map)()

    circuits: List[Any] = []
    kept_samples: List[Sample] = []

    for sample in samples:
        while True:
            try:
                circuit = ansatz(sample.diagram)
                circuits.append(circuit)
                kept_samples.append(sample)
                break
            except KeyError as exc:  # dynamic atomic type encountered
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
                ansatz = _make_ansatz_factory(args, wire_map)()
            except Exception:
                logger.exception(
                    "Failed to convert diagram to circuit; skipping sample: %s",
                    sample.sentence,
                )
                break

    # instantiate the model as the trainer does
    backend_config: dict[str, Any] = {"backend": args.pennylane_backend}
    if args.shots > 0:
        backend_config["shots"] = args.shots

    model = RelationClassifier.from_diagrams(
        circuits,
        n_classes=len(relation_order),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        probabilities=True,
        normalize=True,
        backend_config=backend_config,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    # return parser/rewriter/ansatz factory for later inference
    parser = BobcatParser(verbose="text")
    rewriter = RemoveCupsRewriter()
    ansatz_factory = _make_ansatz_factory(args, wire_map)

    return model, relation_order, parser, rewriter, ansatz_factory


# ---------------------------------------------------------------------------
# inference API
# ---------------------------------------------------------------------------

_nlp = None  # lazy spaCy model loader


def _get_spacy() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:  # model not downloaded
            raise RuntimeError(
                "spaCy English model not found; run `python -m spacy download en_core_web_sm`."
            )
    return _nlp


def extract_entities(sentence: str) -> Tuple[str, str]:
    """Return ``(head, tail)`` entities extracted with a dependency parse.

    A very simple heuristic is used: the first nominal subject
    (``nsubj``/``nsubjpass``) is taken as the head and the first direct or
    prepositional object (``dobj``/``pobj``/``obj``) as the tail.  If the
    parser fails to produce either, fall back to the first/last noun chunk.
    An exception is raised if no reasonable pair can be found.
    """

    doc = _get_spacy()(sentence)
    head = None
    tail = None

    for tok in doc:
        if head is None and tok.dep_ in ("nsubj", "nsubjpass"):
            head = tok.text
        if tail is None and tok.dep_ in ("dobj", "pobj", "obj"):
            tail = tok.text
        if head and tail:
            break

    if not head or not tail:
        chunks = list(doc.noun_chunks)
        if not head and chunks:
            head = chunks[0].text
        if not tail and chunks:
            tail = chunks[-1].text

    if not head or not tail:
        raise ValueError("could not extract both head and tail entities")

    return head, tail


def predict_relation(
    sentence: str,
    model: RelationClassifier,
    parser: BobcatParser,
    rewriter: RemoveCupsRewriter,
    ansatz_factory: Callable[[], IQPAnsatz],
    relation_order: Iterable[str],
) -> Tuple[str, float]:
    """Return the predicted relation and confidence for a single sentence.

    The caller must supply the pipeline objects produced by
    :func:`prepare_pipeline`.
    """

    # convert to diagram then circuit using the same logic as training
    diagram = parser.sentence2diagram(sentence)
    diagram = rewriter(diagram)
    circuit = ansatz_factory()(diagram)

    model.eval()
    with torch.no_grad():
        logits = model([circuit])
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        relation = list(relation_order)[idx]
        confidence = float(probs[idx])
    return relation, confidence


def predict_triple(
    sentence: str,
    model: RelationClassifier,
    parser: BobcatParser,
    rewriter: RemoveCupsRewriter,
    ansatz_factory: Callable[[], IQPAnsatz],
    relation_order: Iterable[str],
) -> Tuple[str, str, str]:
    """Combine entity extraction and relation prediction.

    Returns ``(head, relation, tail)``.  If entity extraction fails the error
    is propagated to the caller.
    """

    head, tail = extract_entities(sentence)
    relation, _ = predict_relation(
        sentence, model, parser, rewriter, ansatz_factory, relation_order
    )
    return head, relation, tail


# ---------------------------------------------------------------------------
# command‑line example / demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Run relation‑extraction inference.")
    cli.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("data/relation_extraction_discocat.csv"),
        help="CSV that was used to train the model (needed to build circuits).",
    )
    cli.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the saved model state_dict (e.g. runs/.../best_model.pt).",
    )
    # a subset of training flags that influence the pipeline
    cli.add_argument("--n-layers", type=int, default=1)
    cli.add_argument("--n-single-qubit-params", type=int, default=3)
    cli.add_argument("--n-qubits-noun", type=int, default=1)
    cli.add_argument("--n-qubits-sentence", type=int, default=3)
    cli.add_argument("--n-qubits-preposition", type=int, default=1)
    cli.add_argument("--n-qubits-other", type=int, default=1)
    cli.add_argument("--pennylane-backend", default="default.qubit")
    cli.add_argument("--shots", type=int, default=0)
    cli.add_argument("--hidden-dim", type=int, default=32)
    cli.add_argument("--dropout", type=float, default=0.1)
    cli.add_argument("--rewrite-log-max-errors", type=int, default=5)
    cli.add_argument(
        "--relation-order",
        nargs="+",
        default=None,
        help="Explicit relation ordering (otherwise inferred from CSV).",
    )
    args = cli.parse_args()

    # basic logger
    logger = logging.getLogger("relation_inference")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    model, relation_order, parser, rewriter, ansatz_factory = prepare_pipeline(
        args.dataset_csv, args.checkpoint, args, logger
    )

    print("Loaded model with relations:", relation_order)

    # demo sentences
    examples = [
        "Alice gave Bob a book.",
        "The cat sat on the mat.",
        "This sentence has no entities.",
    ]

    for sent in examples:
        try:
            triple = predict_triple(
                sent, model, parser, rewriter, ansatz_factory, relation_order
            )
            print(f"{sent!r} -> {triple}")
        except Exception as e:
            print(f"{sent!r} -> error: {e}")
