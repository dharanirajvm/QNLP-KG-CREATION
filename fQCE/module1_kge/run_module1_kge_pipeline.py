#!/usr/bin/env python3
"""End-to-end module1 KGE pipeline for Lambeq/DisCoCat triples."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
FQCE_DIR = THIS_DIR.parent
LLM_SIMPLIFICATION_DIR = FQCE_DIR.parent

DEFAULT_LAMBEQ_JSONL = LLM_SIMPLIFICATION_DIR / "runs" / "lambeq_relation_20260304_193030" / "kg_triples_test.jsonl"
DEFAULT_DATASET_CSV = LLM_SIMPLIFICATION_DIR / "data" / "relation_extraction_discocat_v2.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run module1 KGE: prepare triples -> train -> metrics/viz -> embedding meaning analysis."
    )
    parser.add_argument("--source-mode", choices=["lambeq_generated", "dataset_csv", "both"], default="both")
    parser.add_argument(
        "--lambeq-triples-jsonl",
        type=Path,
        nargs="*",
        default=[DEFAULT_LAMBEQ_JSONL],
        help="JSONL files with head/relation/tail/confidence.",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        nargs="*",
        default=[DEFAULT_DATASET_CSV],
        help="CSV files with triples (relation_extraction_discocat-style).",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Drop triples below this confidence if field exists.")
    parser.add_argument("--deduplicate", action="store_true", help="Deduplicate normalized triples before splitting.")
    parser.add_argument("--max-triples", type=int, default=0, help="If >0, keep only this many triples before splitting.")

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-tag", type=str, default="discocat_lambeq")
    parser.add_argument("--dataset-out-dir", type=Path, default=None, help="Optional fixed output dataset directory.")

    parser.add_argument("--model", choices=["quantum", "complex"], default="quantum")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--negatives-per-positive", type=int, default=32)
    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--q-backend", default="lightning.gpu")
    parser.add_argument("--kappa", type=int, default=1)
    parser.add_argument("--train-samples-per-epoch", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--eval-max-triples", type=int, default=200)
    parser.add_argument("--eval-candidates", type=int, default=2048)
    parser.add_argument("--max-train-triples", type=int, default=0)
    parser.add_argument("--allow-classical", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--existing-run-dir", type=Path, default=None, help="Use this trained run dir instead of training.")
    parser.add_argument("--runs-root", type=Path, default=THIS_DIR / "runs")

    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--skip-meaning-analysis", action="store_true")
    parser.add_argument("--skip-metrics-viz", action="store_true")
    parser.add_argument("--viz-max-entities", type=int, default=2000)
    parser.add_argument("--analysis-anchors", type=int, default=200)
    parser.add_argument("--analysis-nearest-pool", type=int, default=3000)
    parser.add_argument("--analysis-topk", type=int, default=10)
    parser.add_argument("--analysis-max-metric-triples", type=int, default=2000)
    parser.add_argument("--analysis-max-prototype-triples", type=int, default=5000)
    return parser.parse_args()


def resolve_many(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_absolute():
            out.append(p)
        else:
            out.append((LLM_SIMPLIFICATION_DIR / p).resolve())
    return out


def norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _first_nonempty(row: dict[str, str], keys: list[str]) -> str | None:
    for key in keys:
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return None


def _conf_ok(conf_raw: str | None, min_conf: float) -> bool:
    if conf_raw is None:
        return True
    try:
        return float(conf_raw) >= min_conf
    except ValueError:
        return True


def triples_from_jsonl(paths: list[Path], min_conf: float) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    triples: list[tuple[str, str, str]] = []
    stats = {"files_found": 0, "rows_kept": 0, "rows_skipped": 0}
    for path in paths:
        if not path.exists():
            continue
        stats["files_found"] += 1
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["rows_skipped"] += 1
                    continue
                head = obj.get("head")
                rel = obj.get("relation") or obj.get("pred_relation")
                tail = obj.get("tail")
                conf = obj.get("confidence")
                if not head or not rel or not tail or not _conf_ok(str(conf) if conf is not None else None, min_conf):
                    stats["rows_skipped"] += 1
                    continue
                triples.append((norm_text(head), norm_text(rel), norm_text(tail)))
                stats["rows_kept"] += 1
    return triples, stats


def triples_from_csv(paths: list[Path], min_conf: float) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    triples: list[tuple[str, str, str]] = []
    stats = {"files_found": 0, "rows_kept": 0, "rows_skipped": 0}
    head_keys = ["head", "head_entity", "entity1", "entity_1", "entity_1_text", "entity1_text"]
    tail_keys = ["tail", "tail_entity", "entity2", "entity_2", "entity_2_text", "entity2_text"]
    rel_keys = ["relation", "pred_relation", "true_relation", "label"]
    conf_keys = ["confidence", "score", "probability"]

    for path in paths:
        if not path.exists():
            continue
        stats["files_found"] += 1
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                head = _first_nonempty(row, head_keys)
                tail = _first_nonempty(row, tail_keys)
                rel = _first_nonempty(row, rel_keys)
                conf = _first_nonempty(row, conf_keys)
                if not head or not tail or not rel or not _conf_ok(conf, min_conf):
                    stats["rows_skipped"] += 1
                    continue
                triples.append((norm_text(head), norm_text(rel), norm_text(tail)))
                stats["rows_kept"] += 1
    return triples, stats


def split_triples(
    triples: list[tuple[str, str, str]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    if len(triples) < 3:
        raise ValueError("Need at least 3 triples to produce train/valid/test splits.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1).")
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError("--valid-ratio must be in (0, 1).")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("--train-ratio + --valid-ratio must be < 1.0.")

    shuffled = list(triples)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)

    n_train = max(1, int(total * train_ratio))
    n_valid = max(1, int(total * valid_ratio))
    if n_train + n_valid >= total:
        n_valid = max(1, total - n_train - 1)
    n_test = total - n_train - n_valid
    if n_test < 1:
        n_test = 1
        if n_valid > 1:
            n_valid -= 1
        else:
            n_train -= 1

    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]
    return train, valid, test


def write_split(path: Path, triples: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True)


def detect_latest_run(runs_root: Path, model: str, start_time: float) -> Path:
    candidates = [p for p in runs_root.glob(f"{model}_fb15k237_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {runs_root}")
    recent = [p for p in candidates if p.stat().st_mtime >= (start_time - 2.0)]
    target = max(recent or candidates, key=lambda p: p.stat().st_mtime)
    return target.resolve()


def humanize(label: str) -> str:
    return label.strip("/").replace("_", " ").replace("/", " ")


def write_labels_human(snapshot_dir: Path) -> Path:
    entity_path = snapshot_dir / "entity_to_id.json"
    relation_path = snapshot_dir / "relation_to_id.json"
    if not entity_path.exists() or not relation_path.exists():
        raise FileNotFoundError("entity_to_id.json or relation_to_id.json not found in run directory.")

    entity_to_id = json.loads(entity_path.read_text(encoding="utf-8"))
    relation_to_id = json.loads(relation_path.read_text(encoding="utf-8"))

    labels: dict[str, str] = {}
    for raw in entity_to_id.keys():
        labels[raw] = humanize(raw)
    for raw in relation_to_id.keys():
        labels[raw] = humanize(raw)

    out_path = snapshot_dir / "labels_human.json"
    out_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    runs_root = args.runs_root.resolve() if args.runs_root.is_absolute() else (THIS_DIR / args.runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dataset_out_dir is not None:
        dataset_dir = args.dataset_out_dir.resolve() if args.dataset_out_dir.is_absolute() else (THIS_DIR / args.dataset_out_dir).resolve()
    else:
        dataset_dir = (THIS_DIR / "datasets" / f"{args.dataset_tag}_{timestamp}").resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    jsonl_paths = resolve_many(args.lambeq_triples_jsonl)
    csv_paths = resolve_many(args.dataset_csv)

    triples: list[tuple[str, str, str]] = []
    source_stats = {}

    if args.source_mode in ("lambeq_generated", "both"):
        t_jsonl, stats_jsonl = triples_from_jsonl(jsonl_paths, args.min_confidence)
        triples.extend(t_jsonl)
        source_stats["lambeq_generated"] = stats_jsonl
    if args.source_mode in ("dataset_csv", "both"):
        t_csv, stats_csv = triples_from_csv(csv_paths, args.min_confidence)
        triples.extend(t_csv)
        source_stats["dataset_csv"] = stats_csv

    if not triples:
        raise SystemExit(
            "No triples found from selected sources. Check --source-mode and input paths."
        )

    before_dedupe = len(triples)
    if args.deduplicate:
        triples = list(dict.fromkeys(triples))
    if args.max_triples > 0 and len(triples) > args.max_triples:
        random.Random(args.seed).shuffle(triples)
        triples = triples[: args.max_triples]

    train, valid, test = split_triples(
        triples=triples,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    write_split(dataset_dir / "train.txt", train)
    write_split(dataset_dir / "valid.txt", valid)
    write_split(dataset_dir / "test.txt", test)

    rel_counts = Counter(rel for _, rel, _ in triples)
    dataset_summary = {
        "timestamp": timestamp,
        "dataset_dir": str(dataset_dir),
        "source_mode": args.source_mode,
        "source_stats": source_stats,
        "triples_before_dedupe": before_dedupe,
        "triples_after_filters": len(triples),
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "num_unique_entities": len({e for h, _, t in triples for e in (h, t)}),
        "num_relations": len(rel_counts),
        "relation_distribution": dict(sorted(rel_counts.items(), key=lambda x: x[0])),
    }
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    print("Prepared dataset:", dataset_dir)
    print("Split sizes:", dataset_summary["split_sizes"])

    run_dir: Path
    if args.skip_training:
        if args.existing_run_dir is None:
            raise SystemExit("--skip-training requires --existing-run-dir.")
        run_dir = args.existing_run_dir.resolve()
    else:
        train_script = FQCE_DIR / "training_fb15k237.py"
        start_time = time.time()

        cmd = [
            sys.executable,
            str(train_script),
            "--dataset-dir",
            str(dataset_dir),
            "--model",
            args.model,
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--grad-clip",
            str(args.grad_clip),
            "--embedding-dim",
            str(args.embedding_dim),
            "--negatives-per-positive",
            str(args.negatives_per_positive),
            "--num-qubits",
            str(args.num_qubits),
            "--q-backend",
            args.q_backend,
            "--kappa",
            str(args.kappa),
            "--train-samples-per-epoch",
            str(args.train_samples_per_epoch),
            "--eval-every",
            str(args.eval_every),
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--eval-max-triples",
            str(args.eval_max_triples),
            "--eval-candidates",
            str(args.eval_candidates),
            "--max-train-triples",
            str(args.max_train_triples),
            "--output-dir",
            str(runs_root),
            "--seed",
            str(args.seed),
            "--log-level",
            args.log_level,
        ]
        if args.model == "complex":
            cmd.append("--allow-classical")
        run_cmd(cmd)
        run_dir = detect_latest_run(runs_root=runs_root, model=args.model, start_time=start_time)

    labels_path = write_labels_human(run_dir)
    print("Labels file:", labels_path)

    artifacts = {
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "labels_human": str(labels_path),
    }

    if not args.skip_visualization:
        viz_script = THIS_DIR / "visualize_module1_embeddings.py"
        viz_dir = run_dir / "embedding_viz"
        cmd = [
            sys.executable,
            str(viz_script),
            "--snapshot-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--out-dir",
            str(viz_dir),
            "--seed",
            str(args.seed),
            "--max-entities-for-tsne",
            str(args.viz_max_entities),
        ]
        run_cmd(cmd)
        artifacts["embedding_viz_dir"] = str(viz_dir)

    if not args.skip_meaning_analysis:
        analysis_script = THIS_DIR / "analyze_module1_kge_meaning.py"
        analysis_dir = run_dir / "embedding_analysis"
        cmd = [
            sys.executable,
            str(analysis_script),
            "--snapshot-dir",
            str(run_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--out-dir",
            str(analysis_dir),
            "--seed",
            str(args.seed),
            "--anchors",
            str(args.analysis_anchors),
            "--nearest-pool",
            str(args.analysis_nearest_pool),
            "--topk",
            str(args.analysis_topk),
            "--max-metric-triples",
            str(args.analysis_max_metric_triples),
            "--max-prototype-triples",
            str(args.analysis_max_prototype_triples),
        ]
        run_cmd(cmd)
        artifacts["embedding_analysis_dir"] = str(analysis_dir)

    if not args.skip_metrics_viz:
        metrics_script = LLM_SIMPLIFICATION_DIR / "Review-2 outputs" / "code" / "plot_training_metrics.py"
        metrics_dir = run_dir / "metrics_viz"
        cmd = [
            sys.executable,
            str(metrics_script),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(metrics_dir),
            "--smooth",
            "3",
        ]
        run_cmd(cmd)
        artifacts["metrics_viz_dir"] = str(metrics_dir)

    summary = {
        "pipeline": "module1_kge",
        "run_datetime": datetime.now().isoformat(),
        "args": vars(args),
        "dataset_summary": dataset_summary,
        "artifacts": artifacts,
    }
    summary_path = run_dir / "pipeline_summary_module1.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("Pipeline summary:", summary_path)


if __name__ == "__main__":
    main()
