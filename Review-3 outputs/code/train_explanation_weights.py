from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from explanation_utils import (
    DEFAULT_DATASET_DIR,
    DEFAULT_SNAPSHOT_DIR,
    DEFAULT_WEIGHTS_DIR,
    build_explanation_context,
    path_feature_row,
    sample_negative_triples,
    shared_neighbor_feature_row,
)


PATH_FEATURES = [
    "path_reliability",
    "relation_relevance",
    "path_frequency_log",
    "embedding_support",
]

SHARED_FEATURES = [
    "relation_match_strength",
    "hubness_log",
    "embedding_coherence",
]

PATH_CONSTRAINTS = {
    "path_reliability": "nonnegative",
    "relation_relevance": "nonnegative",
    "path_frequency_log": "nonnegative",
    "embedding_support": "nonnegative",
}

SHARED_CONSTRAINTS = {
    "relation_match_strength": "nonnegative",
    "hubness_log": "nonpositive",
    "embedding_coherence": "nonnegative",
}


@dataclass
class Logger:
    log_path: Path

    def __post_init__(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def resolve_device(device_name: str) -> torch.device:
    if device_name != "cuda":
        raise ValueError("This trainer is configured for GPU-only execution. Use --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This explanation-weight training requires a GPU.")
    return torch.device("cuda")


def build_feature_tensors(
    rows: list[dict],
    feature_names: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        return (
            torch.empty((0, len(feature_names)), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )
    features = [
        [float(row.get(name, 0.0)) for name in feature_names]
        for row in rows
    ]
    labels = [float(row["label"]) for row in rows]
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    return x, y


def logistic_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if labels.numel() == 0:
        return {"accuracy": 0.0, "loss": 0.0, "count": 0}
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    accuracy = float((preds == labels).float().mean().item())
    loss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item())
    return {"accuracy": accuracy, "loss": loss, "count": int(labels.numel())}


class ConstrainedLinear(torch.nn.Module):
    def __init__(self, feature_names: list[str], constraints: dict[str, str]) -> None:
        super().__init__()
        self.feature_names = feature_names
        self.constraints = [constraints.get(name, "free") for name in feature_names]
        self.raw_weight = torch.nn.Parameter(torch.zeros(len(feature_names), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def transformed_weight_vector(self) -> torch.Tensor:
        weights: list[torch.Tensor] = []
        for idx, constraint in enumerate(self.constraints):
            raw = self.raw_weight[idx]
            if constraint == "nonnegative":
                weights.append(torch.nn.functional.softplus(raw))
            elif constraint == "nonpositive":
                weights.append(-torch.nn.functional.softplus(raw))
            else:
                weights.append(raw)
        return torch.stack(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_vec = self.transformed_weight_vector()
        return x.matmul(weight_vec) + self.bias


def fit_logistic_regression_gpu(
    rows: list[dict],
    feature_names: list[str],
    *,
    constraints: dict[str, str],
    device: torch.device,
    logger: Logger,
    model_name: str,
    epochs: int = 400,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    log_every: int = 25,
) -> tuple[float, dict[str, float], dict[str, float]]:
    if not rows:
        raise ValueError(f"No training rows were generated for {model_name}.")

    x_train, y_train = build_feature_tensors(rows, feature_names, device)
    linear = ConstrainedLinear(feature_names, constraints).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=learning_rate, weight_decay=l2)
    criterion = torch.nn.BCEWithLogitsLoss()

    logger.log(
        f"[{model_name}] starting GPU training with {x_train.shape[0]} rows, "
        f"{x_train.shape[1]} features, device={device}, epochs={epochs}, lr={learning_rate}, "
        f"weight_decay={l2}, constraints={constraints}"
    )

    final_metrics = {"accuracy": 0.0, "loss": 0.0, "count": int(x_train.shape[0])}
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = linear(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == epochs or epoch % max(1, log_every) == 0:
            with torch.no_grad():
                logits_eval = linear(x_train)
                final_metrics = logistic_metrics(logits_eval, y_train)
            logger.log(
                f"[{model_name}] epoch {epoch}/{epochs} "
                f"train_loss={final_metrics['loss']:.6f} train_acc={final_metrics['accuracy']:.4f}"
            )

    with torch.no_grad():
        learned_weights = linear.transformed_weight_vector().detach().cpu().tolist()
        learned_bias = float(linear.bias.detach().squeeze(0).cpu().item())
        logits_eval = linear(x_train)
        final_metrics = logistic_metrics(logits_eval, y_train)

    weights = {
        feature_name: float(weight_value)
        for feature_name, weight_value in zip(feature_names, learned_weights)
    }
    logger.log(f"[{model_name}] finished constrained training. bias={learned_bias:.6f} weights={weights}")
    return learned_bias, weights, final_metrics


def evaluate_gpu(
    rows: list[dict],
    feature_names: list[str],
    *,
    bias: float,
    weights: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    x_eval, y_eval = build_feature_tensors(rows, feature_names, device)
    if y_eval.numel() == 0:
        return {"accuracy": 0.0, "loss": 0.0, "count": 0}
    weight_vec = torch.tensor(
        [float(weights[name]) for name in feature_names],
        dtype=torch.float32,
        device=device,
    )
    bias_t = torch.tensor(float(bias), dtype=torch.float32, device=device)
    logits = x_eval.matmul(weight_vec) + bias_t
    return logistic_metrics(logits, y_eval)


def collect_path_training_rows(ctx, triples: list[tuple[int, int, int]], label: int) -> list[dict]:
    rows: list[dict] = []
    for h, r, t in triples:
        for r1, mid in ctx.outgoing.get(h, []):
            for r2, end in ctx.outgoing.get(mid, []):
                if end != t:
                    continue
                row = path_feature_row(ctx, head_id=h, relation_id=r, tail_id=t, r1=r1, mid_id=mid, r2=r2)
                row["label"] = label
                rows.append(row)
    return rows


def collect_shared_training_rows(ctx, triples: list[tuple[int, int, int]], label: int) -> list[dict]:
    rows: list[dict] = []
    for h, r, t in triples:
        shared = ctx.undirected_neighbors.get(h, set()) & ctx.undirected_neighbors.get(t, set())
        for nbr in shared:
            row = shared_neighbor_feature_row(ctx, head_id=h, relation_id=r, tail_id=t, nbr_id=nbr)
            row["label"] = label
            rows.append(row)
    return rows


def split_train_valid(rows: list[dict], rng: random.Random, valid_fraction: float) -> tuple[list[dict], list[dict]]:
    rows = list(rows)
    rng.shuffle(rows)
    if not rows:
        return [], []
    valid_size = max(1, int(len(rows) * valid_fraction)) if len(rows) > 1 else 0
    if valid_size >= len(rows):
        valid_size = max(0, len(rows) - 1)
    return rows[valid_size:], rows[:valid_size]


def save_model(
    out_path: Path,
    *,
    feature_names: list[str],
    constraints: dict[str, str],
    bias: float,
    weights: dict[str, float],
    training_summary: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "logistic_regression",
        "feature_names": feature_names,
        "constraints": constraints,
        "bias": bias,
        "weights": weights,
        "training_summary": training_summary,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn explanation and shared-neighbor scoring weights from FB15k data on GPU.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-positive-triples", type=int, default=4000)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--valid-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.weights_dir / "training.log")
    device = resolve_device(args.device)
    cuda_index = torch.cuda.current_device()

    logger.log("Starting explanation-weight training.")
    logger.log(f"Resolved CUDA device: {torch.cuda.get_device_name(cuda_index)} (index={cuda_index})")
    logger.log(f"Arguments: {vars(args)}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    logger.log("Building explanation context from FB15k snapshot and dataset.")
    ctx = build_explanation_context(
        snapshot_dir=args.snapshot_dir,
        dataset_dir=args.dataset_dir,
        weights_dir=args.weights_dir,
    )
    logger.log(
        f"Context ready. train_triples={len(ctx.train_triples)} "
        f"entities={len(ctx.kge.entity_to_id)} relations={len(ctx.kge.relation_to_id)}"
    )

    positives = list(ctx.train_triples)
    rng.shuffle(positives)
    positives = positives[: args.max_positive_triples]
    logger.log(f"Selected {len(positives)} positive triples for explainer-weight training.")

    negatives = sample_negative_triples(
        ctx,
        positives,
        rng=rng,
        negatives_per_positive=args.negatives_per_positive,
    )
    logger.log(
        f"Generated {len(negatives)} negative triples using tail/relation corruption. "
        f"negatives_per_positive={args.negatives_per_positive}"
    )

    logger.log("Collecting path-level training rows.")
    path_rows = collect_path_training_rows(ctx, positives, label=1) + collect_path_training_rows(ctx, negatives, label=0)
    logger.log(f"Collected {len(path_rows)} total path rows.")

    logger.log("Collecting shared-neighbor training rows.")
    shared_rows = collect_shared_training_rows(ctx, positives, label=1) + collect_shared_training_rows(ctx, negatives, label=0)
    logger.log(f"Collected {len(shared_rows)} total shared-neighbor rows.")

    path_train, path_valid = split_train_valid(path_rows, rng, args.valid_fraction)
    shared_train, shared_valid = split_train_valid(shared_rows, rng, args.valid_fraction)
    logger.log(
        f"Split path rows into train={len(path_train)} valid={len(path_valid)}; "
        f"shared rows into train={len(shared_train)} valid={len(shared_valid)}"
    )

    path_bias, path_weights, path_train_metrics = fit_logistic_regression_gpu(
        path_train,
        PATH_FEATURES,
        constraints=PATH_CONSTRAINTS,
        device=device,
        logger=logger,
        model_name="path_model",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        log_every=args.log_every,
    )
    shared_bias, shared_weights, shared_train_metrics = fit_logistic_regression_gpu(
        shared_train,
        SHARED_FEATURES,
        constraints=SHARED_CONSTRAINTS,
        device=device,
        logger=logger,
        model_name="shared_neighbor_model",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        log_every=args.log_every,
    )

    path_summary = {
        "device": str(device),
        "train": path_train_metrics,
        "valid": evaluate_gpu(path_valid, PATH_FEATURES, bias=path_bias, weights=path_weights, device=device),
        "positive_triples_used": len(positives),
        "negative_triples_used": len(negatives),
        "training_rows": len(path_rows),
        "train_rows": len(path_train),
        "valid_rows": len(path_valid),
        "constraints": PATH_CONSTRAINTS,
    }
    shared_summary = {
        "device": str(device),
        "train": shared_train_metrics,
        "valid": evaluate_gpu(shared_valid, SHARED_FEATURES, bias=shared_bias, weights=shared_weights, device=device),
        "positive_triples_used": len(positives),
        "negative_triples_used": len(negatives),
        "training_rows": len(shared_rows),
        "train_rows": len(shared_train),
        "valid_rows": len(shared_valid),
        "constraints": SHARED_CONSTRAINTS,
    }

    logger.log(f"Path model summary: {path_summary}")
    logger.log(f"Shared-neighbor model summary: {shared_summary}")

    save_model(
        args.weights_dir / "path_model.json",
        feature_names=PATH_FEATURES,
        constraints=PATH_CONSTRAINTS,
        bias=path_bias,
        weights=path_weights,
        training_summary=path_summary,
    )
    save_model(
        args.weights_dir / "shared_neighbor_model.json",
        feature_names=SHARED_FEATURES,
        constraints=SHARED_CONSTRAINTS,
        bias=shared_bias,
        weights=shared_weights,
        training_summary=shared_summary,
    )

    summary = {
        "snapshot_dir": str(args.snapshot_dir),
        "dataset_dir": str(args.dataset_dir),
        "weights_dir": str(args.weights_dir),
        "device": str(device),
        "path_model": path_summary,
        "shared_neighbor_model": shared_summary,
        "log_path": str(args.weights_dir / "training.log"),
    }
    summary_path = args.weights_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.log(f"Saved training summary to {summary_path}")
    logger.log(f"Saved learned explanation weights to {args.weights_dir}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
