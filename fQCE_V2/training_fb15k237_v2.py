#!/usr/bin/env python3
"""KGE training V2 with relation-negative sampling for stronger relation prediction."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
FQCE_BASE_DIR = THIS_DIR.parent / "fQCE"
if str(FQCE_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(FQCE_BASE_DIR))

from training_fb15k237 import (  # type: ignore
    ComplexKGE,
    KGData,
    LOGGER,
    QuantumKGE,
    append_jsonl,
    build_filter_maps,
    evaluate_filtered_complex,
    load_fb15k237,
    resolve_device,
    set_seed,
    setup_logging,
    setup_quantum,
    train_complex,
)


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate
    local_candidate = (THIS_DIR / path).resolve()
    return local_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KGE V2 trainer with relation-negative sampling.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("../fQCE/datasets/fb15k237"))
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--model", choices=["complex", "quantum"], default="quantum")
    parser.add_argument("--allow-classical", action="store_true")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--negatives-per-positive", type=int, default=32)

    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--q-backend", default="lightning.gpu")
    parser.add_argument("--kappa", type=int, default=1)
    parser.add_argument("--train-samples-per-epoch", type=int, default=0)
    parser.add_argument("--entity-negatives-per-positive", type=int, default=1)
    parser.add_argument("--relation-negatives-per-positive", type=int, default=1)
    parser.add_argument("--train-log-every-batches", type=int, default=10, help="Log detailed batch progress every N batches. 0 disables periodic batch logs.")
    parser.add_argument("--log-first-n-batches", type=int, default=3, help="Always log details for the first N batches of each epoch.")
    parser.add_argument("--log-sampled-triples-per-epoch", type=int, default=3, help="Log this many sampled training triples at the start of each epoch.")

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--eval-max-triples", type=int, default=200)
    parser.add_argument("--eval-protocol", choices=["sampled", "exact"], default="sampled")
    parser.add_argument("--eval-candidates", type=int, default=2048)
    parser.add_argument("--eval-relation-candidates", type=int, default=0, help="0 means use all relations.")
    parser.add_argument("--relation-eval-weight", type=float, default=0.5, help="Weight of relation MRR in checkpoint selection.")

    parser.add_argument("--max-train-triples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "runs_kge_v2")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def build_relation_filter_map(all_true: set[tuple[int, int, int]]) -> dict[tuple[int, int], set[int]]:
    rels: dict[tuple[int, int], set[int]] = {}
    for h, r, t in all_true:
        rels.setdefault((h, t), set()).add(r)
    return rels


def sample_negative_tail(true_tail: int, num_entities: int) -> int:
    neg_t = random.randint(0, num_entities - 1)
    while neg_t == true_tail:
        neg_t = random.randint(0, num_entities - 1)
    return neg_t


def sample_negative_relation(true_relation: int, num_relations: int) -> int:
    neg_r = random.randint(0, num_relations - 1)
    while neg_r == true_relation:
        neg_r = random.randint(0, num_relations - 1)
    return neg_r


def log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    ordered = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items() if key != "n")
    LOGGER.info("%s | %s | n=%.0f", prefix, ordered, metrics.get("n", 0.0))


def format_triple(h: int, r: int, t: int, id_to_entity: dict[int, str], id_to_relation: dict[int, str]) -> str:
    return f"({id_to_entity[h]}, {id_to_relation[r]}, {id_to_entity[t]})"


def evaluate_relation_prediction_quantum(
    model: QuantumKGE,
    triples: list[tuple[int, int, int]],
    relation_filter: dict[tuple[int, int], set[int]],
    num_relations: int,
    max_triples: int,
    eval_relation_candidates: int,
) -> dict[str, float]:
    if max_triples > 0 and len(triples) > max_triples:
        triples = random.sample(triples, max_triples)
    if not triples:
        return {"mr": 0.0, "mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0, "n": 0.0}

    ranks: list[int] = []
    for h, r, t in triples:
        if eval_relation_candidates > 0 and eval_relation_candidates < num_relations:
            rel_cands = {r}
            while len(rel_cands) < min(eval_relation_candidates, num_relations):
                rel_cands.add(random.randint(0, num_relations - 1))
        else:
            rel_cands = set(range(num_relations))

        scores: dict[int, float] = {}
        for cand_r in rel_cands:
            scores[cand_r] = float(model.score(h, cand_r, t).item())

        true_score = scores[r]
        filtered = set(relation_filter.get((h, t), set()))
        filtered.discard(r)
        rank = 1 + sum(
            1 for cand_r, score in scores.items() if cand_r != r and cand_r not in filtered and score > true_score
        )
        ranks.append(rank)

    n = len(ranks)
    return {
        "mr": sum(ranks) / n,
        "mrr": sum(1.0 / x for x in ranks) / n,
        "hits@1": sum(1 for x in ranks if x <= 1) / n,
        "hits@3": sum(1 for x in ranks if x <= 3) / n,
        "hits@10": sum(1 for x in ranks if x <= 10) / n,
        "n": float(n),
    }


def evaluate_filtered_quantum_v2(
    model: QuantumKGE,
    triples: list[tuple[int, int, int]],
    tails_filter: dict[tuple[int, int], set[int]],
    heads_filter: dict[tuple[int, int], set[int]],
    relation_filter: dict[tuple[int, int], set[int]],
    num_entities: int,
    num_relations: int,
    max_triples: int,
    eval_candidates: int,
    eval_relation_candidates: int,
) -> dict[str, float]:
    entity_metrics = _evaluate_entity_prediction_quantum(
        model=model,
        triples=triples,
        tails_filter=tails_filter,
        heads_filter=heads_filter,
        num_entities=num_entities,
        max_triples=max_triples,
        eval_candidates=eval_candidates,
    )
    relation_metrics = evaluate_relation_prediction_quantum(
        model=model,
        triples=triples,
        relation_filter=relation_filter,
        num_relations=num_relations,
        max_triples=max_triples,
        eval_relation_candidates=eval_relation_candidates,
    )
    out = {f"entity_{k}": v for k, v in entity_metrics.items()}
    out.update({f"relation_{k}": v for k, v in relation_metrics.items()})
    return out


def _evaluate_entity_prediction_quantum(
    model: QuantumKGE,
    triples: list[tuple[int, int, int]],
    tails_filter: dict[tuple[int, int], set[int]],
    heads_filter: dict[tuple[int, int], set[int]],
    num_entities: int,
    max_triples: int,
    eval_candidates: int,
) -> dict[str, float]:
    if max_triples > 0 and len(triples) > max_triples:
        triples = random.sample(triples, max_triples)
    if not triples:
        return {"mr": 0.0, "mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0, "n": 0.0}

    cached_entities = model.cached_entity_states()
    ranks: list[int] = []

    for h, r, t in triples:
        tail_cands = {t}
        head_cands = {h}
        while len(tail_cands) < min(eval_candidates, num_entities):
            tail_cands.add(random.randint(0, num_entities - 1))
        while len(head_cands) < min(eval_candidates, num_entities):
            head_cands.add(random.randint(0, num_entities - 1))

        sp_state = model.relation_subject_state(h, r)
        tail_scores = {c: float(torch.real(torch.vdot(cached_entities[c], sp_state)).item()) for c in tail_cands}

        o_state_conj = torch.conj(cached_entities[t])
        head_scores = {}
        for c in head_cands:
            sp_c = model.relation_subject_state(c, r)
            head_scores[c] = float(torch.real(torch.vdot(o_state_conj, sp_c)).item())

        tail_true = tail_scores[t]
        tail_filtered = tails_filter.get((h, r), set())
        tail_rank = 1 + sum(1 for c, s in tail_scores.items() if c != t and c not in tail_filtered and s > tail_true)

        head_true = head_scores[h]
        head_filtered = heads_filter.get((r, t), set())
        head_rank = 1 + sum(1 for c, s in head_scores.items() if c != h and c not in head_filtered and s > head_true)

        ranks.append(tail_rank)
        ranks.append(head_rank)

    n = len(ranks)
    return {
        "mr": sum(ranks) / n,
        "mrr": sum(1.0 / x for x in ranks) / n,
        "hits@1": sum(1 for x in ranks if x <= 1) / n,
        "hits@3": sum(1 for x in ranks if x <= 3) / n,
        "hits@10": sum(1 for x in ranks if x <= 10) / n,
        "n": float(n),
    }


def train_quantum_v2(args: argparse.Namespace, data: KGData, run_dir: Path) -> None:
    setup_quantum(args.num_qubits, args.q_backend)
    model = QuantumKGE(data.num_entities, data.num_relations, args.num_qubits)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    id_to_entity = {v: k for k, v in data.entity_to_id.items()}
    id_to_relation = {v: k for k, v in data.relation_to_id.items()}

    train = data.train
    all_true = set(data.train) | set(data.val) | set(data.test)
    tails_filter, heads_filter = build_filter_maps(all_true)
    relation_filter = build_relation_filter_map(all_true)

    best_selection_score = float("-inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    history_path = run_dir / "metrics_history.jsonl"

    LOGGER.info(
        "Quantum V2 setup | num_qubits=%d backend=%s lr=%g grad_clip=%g kappa=%d entity_negs=%d relation_negs=%d",
        args.num_qubits,
        args.q_backend,
        args.learning_rate,
        args.grad_clip,
        args.kappa,
        args.entity_negatives_per_positive,
        args.relation_negatives_per_positive,
    )
    LOGGER.info(
        "Evaluation setup | eval_every=%d eval_max_triples=%d eval_candidates=%d eval_relation_candidates=%d relation_eval_weight=%.3f",
        args.eval_every,
        args.eval_max_triples,
        args.eval_candidates,
        args.eval_relation_candidates,
        args.relation_eval_weight,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.train_samples_per_epoch > 0 and args.train_samples_per_epoch < len(train):
            epoch_train = random.sample(train, args.train_samples_per_epoch)
        else:
            epoch_train = list(train)
        random.shuffle(epoch_train)

        LOGGER.info(
            "Epoch %03d start | sampled_train_triples=%d from_total_train=%d",
            epoch,
            len(epoch_train),
            len(train),
        )
        for idx, triple in enumerate(epoch_train[: max(0, args.log_sampled_triples_per_epoch)], start=1):
            LOGGER.info(
                "Epoch %03d sample_triple[%d]=%s",
                epoch,
                idx,
                format_triple(triple[0], triple[1], triple[2], id_to_entity, id_to_relation),
            )

        loss_sum = 0.0
        tail_pair_acc_sum = 0.0
        relation_pair_acc_sum = 0.0
        batches = 0

        loop = tqdm(epoch_train, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for h, r, t in loop:
            optimizer.zero_grad()

            sp = model.relation_subject_state(h, r)
            pos_e = model.entity_state(t)
            pos = torch.real(torch.vdot(pos_e, sp))

            tail_neg_scores = []
            for _ in range(max(1, args.entity_negatives_per_positive)):
                neg_t = sample_negative_tail(t, data.num_entities)
                neg_e = model.entity_state(neg_t)
                tail_neg_scores.append(torch.real(torch.vdot(neg_e, sp)))

            relation_neg_scores = []
            for _ in range(max(1, args.relation_negatives_per_positive)):
                neg_r = sample_negative_relation(r, data.num_relations)
                neg_sp = model.relation_subject_state(h, neg_r)
                relation_neg_scores.append(torch.real(torch.vdot(pos_e, neg_sp)))

            loss_terms = [(1.0 - pos) ** (2 * args.kappa)]
            loss_terms.extend([(-1.0 - neg_score) ** (2 * args.kappa) for neg_score in tail_neg_scores])
            loss_terms.extend([(-1.0 - neg_score) ** (2 * args.kappa) for neg_score in relation_neg_scores])
            loss = torch.stack(loss_terms).mean()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            tail_pair = sum(float(pos.item()) > float(neg.item()) for neg in tail_neg_scores) / max(1, len(tail_neg_scores))
            relation_pair = sum(float(pos.item()) > float(neg.item()) for neg in relation_neg_scores) / max(1, len(relation_neg_scores))

            loss_sum += float(loss.item())
            tail_pair_acc_sum += tail_pair
            relation_pair_acc_sum += relation_pair
            batches += 1

            should_log_batch = False
            if batches <= max(0, args.log_first_n_batches):
                should_log_batch = True
            elif args.train_log_every_batches > 0 and batches % args.train_log_every_batches == 0:
                should_log_batch = True
            if should_log_batch:
                LOGGER.info(
                    "Epoch %03d batch %04d/%04d | triple=%s | pos=%.4f tail_neg_mean=%.4f rel_neg_mean=%.4f loss=%.4f tail_pair=%.4f rel_pair=%.4f",
                    epoch,
                    batches,
                    len(epoch_train),
                    format_triple(h, r, t, id_to_entity, id_to_relation),
                    float(pos.item()),
                    sum(float(x.item()) for x in tail_neg_scores) / max(1, len(tail_neg_scores)),
                    sum(float(x.item()) for x in relation_neg_scores) / max(1, len(relation_neg_scores)),
                    float(loss.item()),
                    tail_pair,
                    relation_pair,
                )

            loop.set_postfix(
                train_loss=loss_sum / batches,
                tail_pair_acc=tail_pair_acc_sum / batches,
                relation_pair_acc=relation_pair_acc_sum / batches,
            )

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(1, batches),
            "train_pair_acc": (tail_pair_acc_sum + relation_pair_acc_sum) / max(1, 2 * batches),
            "train_tail_pair_acc": tail_pair_acc_sum / max(1, batches),
            "train_relation_pair_acc": relation_pair_acc_sum / max(1, batches),
            "epoch_seconds": time.time() - t0,
            "train_samples": float(len(epoch_train)),
            "entity_negatives_per_positive": float(args.entity_negatives_per_positive),
            "relation_negatives_per_positive": float(args.relation_negatives_per_positive),
        }
        LOGGER.info(
            "Epoch %03d end | train_loss=%.4f train_pair_acc=%.4f tail_pair_acc=%.4f relation_pair_acc=%.4f epoch_seconds=%.2f",
            epoch,
            row["train_loss"],
            row["train_pair_acc"],
            row["train_tail_pair_acc"],
            row["train_relation_pair_acc"],
            row["epoch_seconds"],
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            LOGGER.info("Epoch %03d evaluation start | split=validation", epoch)
            val_metrics = evaluate_filtered_quantum_v2(
                model=model,
                triples=data.val,
                tails_filter=tails_filter,
                heads_filter=heads_filter,
                relation_filter=relation_filter,
                num_entities=data.num_entities,
                num_relations=data.num_relations,
                max_triples=args.eval_max_triples,
                eval_candidates=args.eval_candidates,
                eval_relation_candidates=args.eval_relation_candidates,
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["val_selection_score"] = val_metrics["entity_hits@3"] + args.relation_eval_weight * val_metrics["relation_mrr"]

            LOGGER.info(
                "[Quantum V2] Epoch %03d | loss=%.4f tail_pair=%.4f rel_pair=%.4f | val_entity_hits@3=%.4f val_relation_mrr=%.4f selection_score=%.4f",
                epoch,
                row["train_loss"],
                row["train_tail_pair_acc"],
                row["train_relation_pair_acc"],
                val_metrics["entity_hits@3"],
                val_metrics["relation_mrr"],
                row["val_selection_score"],
            )
            log_metrics(f"Epoch {epoch:03d} validation entity-ranking", {k.replace('entity_', ''): v for k, v in val_metrics.items() if k.startswith("entity_")})
            log_metrics(f"Epoch {epoch:03d} validation relation-ranking", {k.replace('relation_', ''): v for k, v in val_metrics.items() if k.startswith("relation_")})

            if row["val_selection_score"] > best_selection_score:
                prev_best = best_selection_score
                best_selection_score = row["val_selection_score"]
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, run_dir / "best_model.pt")
                stale = 0
                LOGGER.info(
                    "Checkpoint update | epoch=%03d improved selection_score from %.4f to %.4f | saved=%s",
                    epoch,
                    prev_best,
                    best_selection_score,
                    run_dir / "best_model.pt",
                )
            else:
                stale += 1
                LOGGER.info(
                    "No checkpoint update | epoch=%03d selection_score=%.4f best=%.4f stale=%d/%d",
                    epoch,
                    row["val_selection_score"],
                    best_selection_score,
                    stale,
                    args.early_stop_patience,
                )

            if stale >= args.early_stop_patience:
                LOGGER.info("[Quantum V2] Early stopping at epoch %d", epoch)
                history.append(row)
                append_jsonl(history_path, {"timestamp": datetime.now().isoformat(), **row})
                break

        history.append(row)
        append_jsonl(history_path, {"timestamp": datetime.now().isoformat(), **row})
        LOGGER.info("Epoch %03d metrics appended to %s", epoch, history_path)

    if best_state is not None:
        model.load_state_dict(best_state)
        LOGGER.info("Loaded best checkpoint from epoch %03d for final testing.", best_epoch)
    else:
        LOGGER.warning("No best checkpoint was stored before test evaluation; using current model state.")

    LOGGER.info("Final evaluation start | split=test")
    test_metrics = evaluate_filtered_quantum_v2(
        model=model,
        triples=data.test,
        tails_filter=tails_filter,
        heads_filter=heads_filter,
        relation_filter=relation_filter,
        num_entities=data.num_entities,
        num_relations=data.num_relations,
        max_triples=args.eval_max_triples,
        eval_candidates=args.eval_candidates,
        eval_relation_candidates=args.eval_relation_candidates,
    )
    log_metrics("Test entity-ranking", {k.replace('entity_', ''): v for k, v in test_metrics.items() if k.startswith("entity_")})
    log_metrics("Test relation-ranking", {k.replace('relation_', ''): v for k, v in test_metrics.items() if k.startswith("relation_")})

    torch.save(model.state_dict(), run_dir / "last_model.pt")
    LOGGER.info("Saved final model state to %s", run_dir / "last_model.pt")
    summary = {
        "model": "quantum_v2",
        "best_epoch": best_epoch,
        "best_val_selection_score": best_selection_score,
        "test_filtered_ranking": test_metrics,
        "history": history,
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
        "n_train": len(data.train),
        "n_val": len(data.val),
        "n_test": len(data.test),
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote run summary to %s", run_dir / "metrics_summary.json")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.model == "complex" and not args.allow_classical:
        raise SystemExit(
            "Classical mode is disabled by default. Use --model quantum or pass --allow-classical explicitly."
        )

    dataset_dir = resolve_input_path(args.dataset_dir)
    output_dir = resolve_input_path(args.output_dir)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{args.model}_v2_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir, args.log_level)
    LOGGER.info("===== fQCE V2 run start =====")
    LOGGER.info("Timestamp=%s", timestamp)
    LOGGER.info("Resolved dataset_dir=%s", dataset_dir)
    LOGGER.info("Resolved output_dir=%s", output_dir)
    LOGGER.info("Resolved run_dir=%s", run_dir)
    LOGGER.info("Arguments=%s", {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})

    LOGGER.info("Stage: load_dataset")
    data = load_fb15k237(dataset_dir=dataset_dir, download=args.download, max_train_triples=args.max_train_triples)
    device = resolve_device(args.device)
    LOGGER.info("Stage complete: load_dataset")

    LOGGER.info("Training start | model=%s device=%s", args.model, device)
    if args.model == "quantum":
        LOGGER.info("Quantum backend requested: %s", args.q_backend)
        LOGGER.info("V2 feature enabled: relation-negative sampling + filtered relation ranking evaluation")
    else:
        LOGGER.info("Using classical training path from base implementation.")
    LOGGER.info(
        "dataset_dir=%s entities=%d relations=%d train=%d val=%d test=%d",
        dataset_dir,
        data.num_entities,
        data.num_relations,
        len(data.train),
        len(data.val),
        len(data.test),
    )
    LOGGER.info("run_dir=%s", run_dir)

    config = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "resolved_device": str(device),
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
        "n_train": len(data.train),
        "n_val": len(data.val),
        "n_test": len(data.test),
        "version": "v2",
        "features": {
            "relation_negative_sampling": args.model == "quantum",
            "filtered_relation_prediction_eval": args.model == "quantum",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "entity_to_id.json").write_text(json.dumps(data.entity_to_id, indent=2), encoding="utf-8")
    (run_dir / "relation_to_id.json").write_text(json.dumps(data.relation_to_id, indent=2), encoding="utf-8")
    LOGGER.info("Saved run config and id maps into %s", run_dir)

    if args.model == "complex":
        train_complex(args=args, data=data, run_dir=run_dir, device=device)
    else:
        train_quantum_v2(args=args, data=data, run_dir=run_dir)
    LOGGER.info("===== fQCE V2 run end =====")


if __name__ == "__main__":
    main()
