import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pennylane as qml
import torch
import torch.nn as nn
from tqdm import tqdm


# ===============================
# Data + Config
# ===============================


@dataclass
class KGData:
    triples: list[tuple[int, int, int]]
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fQCE embeddings for KG triples.")
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("../data/relation_extraction_discocat_v2.csv"),
        help="CSV with columns: sentence,entity1,entity2,relation",
    )
    parser.add_argument("--num-qubits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep 1 unless you vectorize QNodes.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--margin", type=float, default=1.0, help="Pairwise ranking margin.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--eval-rank-samples", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=Path("runs_fqce"))
    parser.add_argument(
        "--max-triples",
        type=int,
        default=0,
        help="If >0, use only the first N triples after loading (for faster experiments).",
    )
    parser.add_argument("--use-toy", action="store_true", help="Use synthetic toy triples.")
    parser.add_argument("--toy-entities", type=int, default=20)
    parser.add_argument("--toy-relations", type=int, default=5)
    parser.add_argument("--toy-triples", type=int, default=300)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_kg_from_csv(dataset_csv: Path) -> KGData:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    entity_to_id: dict[str, int] = {}
    relation_to_id: dict[str, int] = {}
    triples: list[tuple[int, int, int]] = []

    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"entity1", "entity2", "relation"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV must include columns {sorted(required)}; got {reader.fieldnames}"
            )

        for row in reader:
            h = str(row["entity1"]).strip().lower()
            t = str(row["entity2"]).strip().lower()
            r = str(row["relation"]).strip().lower()
            if not h or not t or not r:
                continue

            if h not in entity_to_id:
                entity_to_id[h] = len(entity_to_id)
            if t not in entity_to_id:
                entity_to_id[t] = len(entity_to_id)
            if r not in relation_to_id:
                relation_to_id[r] = len(relation_to_id)

            triples.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))

    if not triples:
        raise ValueError("No valid triples loaded from CSV.")

    return KGData(
        triples=triples,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )


def make_toy_data(num_entities: int, num_relations: int, num_triples: int) -> KGData:
    triples = [
        (
            random.randint(0, num_entities - 1),
            random.randint(0, num_relations - 1),
            random.randint(0, num_entities - 1),
        )
        for _ in range(num_triples)
    ]
    entity_to_id = {f"e{i}": i for i in range(num_entities)}
    relation_to_id = {f"r{i}": i for i in range(num_relations)}
    return KGData(triples=triples, entity_to_id=entity_to_id, relation_to_id=relation_to_id)


def split_triples(
    triples: list[tuple[int, int, int]], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    idxs = list(range(len(triples)))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)

    n = len(idxs)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train = [triples[i] for i in idxs[:n_train]]
    val = [triples[i] for i in idxs[n_train : n_train + n_val]]
    test = [triples[i] for i in idxs[n_train + n_val :]]
    return train, val, test


# ===============================
# Quantum Model
# ===============================


class FQCEContext:
    num_qubits: int = 3
    device = None


def setup_quantum(num_qubits: int) -> None:
    FQCEContext.num_qubits = num_qubits
    FQCEContext.device = qml.device("default.qubit", wires=num_qubits)


def circuit_block(params: torch.Tensor) -> None:
    num_qubits = FQCEContext.num_qubits

    for q in range(num_qubits):
        qml.Rot(params[0, q, 0], params[0, q, 1], params[0, q, 2], wires=q)

    # Avoid invalid same-wire control/target pairs on small qubit counts.
    for block_idx, offset in enumerate((1, 2, 3), start=1):
        for q in range(num_qubits):
            target = (q + offset) % num_qubits
            if target == q:
                continue
            qml.CRot(
                params[block_idx, q, 0],
                params[block_idx, q, 1],
                params[block_idx, q, 2],
                wires=[q, target],
            )


def build_qnodes():
    @qml.qnode(FQCEContext.device, interface="torch", diff_method="backprop")
    def fqce_sp_circuit(s_params, p_params):
        for q in range(FQCEContext.num_qubits):
            qml.Hadamard(wires=q)
        circuit_block(s_params)
        circuit_block(p_params)
        return qml.state()

    @qml.qnode(FQCEContext.device, interface="torch", diff_method="backprop")
    def fqce_o_circuit(o_params):
        for q in range(FQCEContext.num_qubits):
            qml.Hadamard(wires=q)
        circuit_block(o_params)
        return qml.state()

    return fqce_sp_circuit, fqce_o_circuit


class FQCE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits

        self.entity_params = nn.Parameter(torch.randn(num_entities, 4, num_qubits, 3) * 0.1)
        self.relation_params = nn.Parameter(torch.randn(num_relations, 4, num_qubits, 3) * 0.1)

        self.fqce_sp_circuit, self.fqce_o_circuit = build_qnodes()

    def score(self, s: int, p: int, o: int) -> torch.Tensor:
        s_params = self.entity_params[s]
        p_params = self.relation_params[p]
        o_params = self.entity_params[o]

        sp_state = self.fqce_sp_circuit(s_params, p_params)
        o_state = self.fqce_o_circuit(o_params)

        # Hermitian inner product to respect complex-valued statevectors.
        score = torch.real(torch.vdot(sp_state, o_state))
        return score


# ===============================
# Dataset
# ===============================


class KGDataset(torch.utils.data.Dataset):
    def __init__(self, triples: list[tuple[int, int, int]], num_entities: int):
        self.triples = triples
        self.num_entities = num_entities

    def __len__(self) -> int:
        return len(self.triples)

    def corrupt(self, s: int, r: int, o: int) -> tuple[int, int, int]:
        if random.random() < 0.5:
            s_new = random.randint(0, self.num_entities - 1)
            while s_new == s:
                s_new = random.randint(0, self.num_entities - 1)
            return s_new, r, o
        o_new = random.randint(0, self.num_entities - 1)
        while o_new == o:
            o_new = random.randint(0, self.num_entities - 1)
        return s, r, o_new

    def __getitem__(self, idx: int):
        s, r, o = self.triples[idx]
        neg = self.corrupt(s, r, o)
        return (s, r, o), neg


# ===============================
# Metrics + Train
# ===============================


@torch.no_grad()
def evaluate_pairwise(model: FQCE, loader) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    correct = 0
    sum_margin = 0.0
    sum_pos = 0.0
    sum_neg = 0.0

    for pos, neg in loader:
        s, p, o = pos
        s_neg, p_neg, o_neg = neg

        s = int(s.item())
        p = int(p.item())
        o = int(o.item())
        s_neg = int(s_neg.item())
        p_neg = int(p_neg.item())
        o_neg = int(o_neg.item())

        pos_score = model.score(s, p, o)
        neg_score = model.score(s_neg, p_neg, o_neg)

        loss = torch.nn.functional.softplus(1.0 - (pos_score - neg_score))

        total_loss += float(loss.item())
        total_pairs += 1
        sum_margin += float((pos_score - neg_score).item())
        sum_pos += float(pos_score.item())
        sum_neg += float(neg_score.item())
        correct += int(pos_score.item() > neg_score.item())

    if total_pairs == 0:
        return {
            "loss": 0.0,
            "pairwise_acc": 0.0,
            "avg_margin": 0.0,
            "avg_pos_score": 0.0,
            "avg_neg_score": 0.0,
        }

    return {
        "loss": total_loss / total_pairs,
        "pairwise_acc": correct / total_pairs,
        "avg_margin": sum_margin / total_pairs,
        "avg_pos_score": sum_pos / total_pairs,
        "avg_neg_score": sum_neg / total_pairs,
    }


@torch.no_grad()
def evaluate_ranking(
    model: FQCE,
    triples: Iterable[tuple[int, int, int]],
    num_entities: int,
    max_samples: int,
) -> dict[str, float]:
    model.eval()
    triples_list = list(triples)
    if not triples_list:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0, "n": 0.0}

    if max_samples > 0 and len(triples_list) > max_samples:
        triples_list = random.sample(triples_list, max_samples)

    rr_sum = 0.0
    h1 = 0
    h3 = 0
    h10 = 0

    for s, p, o in triples_list:
        scores = [float(model.score(s, p, cand).item()) for cand in range(num_entities)]
        target = scores[o]
        better = sum(1 for val in scores if val > target)
        rank = better + 1

        rr_sum += 1.0 / rank
        h1 += int(rank <= 1)
        h3 += int(rank <= 3)
        h10 += int(rank <= 10)

    n = len(triples_list)
    return {
        "mrr": rr_sum / n,
        "hits@1": h1 / n,
        "hits@3": h3 / n,
        "hits@10": h10 / n,
        "n": float(n),
    }


def train(
    model: FQCE,
    train_loader,
    val_loader,
    test_loader,
    train_triples,
    val_triples,
    test_triples,
    num_entities: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    margin: float,
    grad_clip: float,
    early_stop_patience: int,
    eval_rank_samples: int,
    run_dir: Path,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_mrr = -1.0
    best_state = None
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_pairs = 0
        correct = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for pos, neg in loop:
            s, p, o = pos
            s_neg, p_neg, o_neg = neg
            s = int(s.item())
            p = int(p.item())
            o = int(o.item())
            s_neg = int(s_neg.item())
            p_neg = int(p_neg.item())
            o_neg = int(o_neg.item())

            optimizer.zero_grad()

            pos_score = model.score(s, p, o)
            neg_score = model.score(s_neg, p_neg, o_neg)

            loss = torch.nn.functional.softplus(margin - (pos_score - neg_score))
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += float(loss.item())
            total_pairs += 1
            correct += int(pos_score.item() > neg_score.item())
            loop.set_postfix(
                train_loss=total_loss / max(1, total_pairs),
                pair_acc=correct / max(1, total_pairs),
            )

        train_metrics = {
            "train_loss": total_loss / max(1, total_pairs),
            "train_pairwise_acc": correct / max(1, total_pairs),
        }
        val_pairwise = evaluate_pairwise(model, val_loader)
        val_rank = evaluate_ranking(model, val_triples, num_entities, eval_rank_samples)

        row = {
            "epoch": float(epoch),
            **train_metrics,
            **{f"val_{k}": v for k, v in val_pairwise.items()},
            **{f"val_{k}": v for k, v in val_rank.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"train_pair_acc={train_metrics['train_pairwise_acc']:.4f} | "
            f"val_loss={val_pairwise['loss']:.4f} "
            f"val_pair_acc={val_pairwise['pairwise_acc']:.4f} "
            f"val_mrr={val_rank['mrr']:.4f} "
            f"val_hits@10={val_rank['hits@10']:.4f}"
        )

        if val_rank["mrr"] > best_val_mrr:
            best_val_mrr = val_rank["mrr"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no val MRR improvement for {stale_epochs} epochs).")
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    test_pairwise = evaluate_pairwise(model, test_loader)
    test_rank = evaluate_ranking(model, test_triples, num_entities, eval_rank_samples)

    print("\nBest model summary")
    print(f"best_epoch={best_epoch} best_val_mrr={best_val_mrr:.4f}")
    print(
        f"test_loss={test_pairwise['loss']:.4f} "
        f"test_pair_acc={test_pairwise['pairwise_acc']:.4f} "
        f"test_mrr={test_rank['mrr']:.4f} "
        f"test_hits@1={test_rank['hits@1']:.4f} "
        f"test_hits@3={test_rank['hits@3']:.4f} "
        f"test_hits@10={test_rank['hits@10']:.4f}"
    )

    torch.save(model.state_dict(), run_dir / "best_model.pt")

    summary = {
        "best_epoch": best_epoch,
        "best_val_mrr": best_val_mrr,
        "test_pairwise": test_pairwise,
        "test_ranking": test_rank,
        "history": history,
        "n_train": len(train_triples),
        "n_val": len(val_triples),
        "n_test": len(test_triples),
        "num_entities": num_entities,
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_quantum(args.num_qubits)

    if args.use_toy:
        data = make_toy_data(args.toy_entities, args.toy_relations, args.toy_triples)
    else:
        csv_path = args.dataset_csv
        if not csv_path.is_absolute():
            csv_path = (Path(__file__).resolve().parent / csv_path).resolve()
        data = load_kg_from_csv(csv_path)

    if args.max_triples > 0 and len(data.triples) > args.max_triples:
        data = KGData(
            triples=data.triples[: args.max_triples],
            entity_to_id=data.entity_to_id,
            relation_to_id=data.relation_to_id,
        )

    train_triples, val_triples, test_triples = split_triples(
        data.triples, args.val_ratio, args.test_ratio, args.seed
    )

    train_loader = torch.utils.data.DataLoader(
        KGDataset(train_triples, data.num_entities),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        KGDataset(val_triples, data.num_entities),
        batch_size=1,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        KGDataset(test_triples, data.num_entities),
        batch_size=1,
        shuffle=False,
    )

    model = FQCE(data.num_entities, data.num_relations, args.num_qubits)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"fqce_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    serializable_args = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    config = {
        "args": serializable_args,
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("fQCE training start")
    print(
        f"entities={data.num_entities} relations={data.num_relations} "
        f"train={len(train_triples)} val={len(val_triples)} test={len(test_triples)}"
    )
    print(f"run_dir={run_dir}")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_triples=train_triples,
        val_triples=val_triples,
        test_triples=test_triples,
        num_entities=data.num_entities,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.margin,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        eval_rank_samples=args.eval_rank_samples,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()


#old implementation that was successfully trained on the toy dataset. Use it for our end-end pipeline later when needed!!