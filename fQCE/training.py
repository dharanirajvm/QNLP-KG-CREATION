import argparse
import csv
import json
import logging
import random
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pennylane as qml
import torch
import torch.nn as nn
from tqdm import tqdm


KINSHIP_URLS = {
    "train": "https://raw.githubusercontent.com/pykeen/pykeen/master/src/pykeen/datasets/kinships/train.txt",
    "valid": "https://raw.githubusercontent.com/pykeen/pykeen/master/src/pykeen/datasets/kinships/valid.txt",
    "test": "https://raw.githubusercontent.com/pykeen/pykeen/master/src/pykeen/datasets/kinships/test.txt",
}


@dataclass
class KGData:
    train: list[tuple[int, int, int]]
    val: list[tuple[int, int, int]]
    test: list[tuple[int, int, int]]
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)


LOGGER = logging.getLogger("fqce")


def setup_logging(run_dir: Path, level: str) -> None:
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_h = logging.StreamHandler()
    stream_h.setFormatter(formatter)
    LOGGER.addHandler(stream_h)

    file_h = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    file_h.setFormatter(formatter)
    LOGGER.addHandler(file_h)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fQCE training aligned to VQC-KGE methodology.")

    parser.add_argument("--dataset", choices=["kinship", "csv", "toy"], default="kinship")
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("../data/relation_extraction_discocat_v2.csv"),
        help="Used when --dataset csv.",
    )
    parser.add_argument(
        "--kinship-dir",
        type=Path,
        default=Path("datasets/kinship"),
        help="Used when --dataset kinship.",
    )
    parser.add_argument(
        "--download-kinship",
        action="store_true",
        help="Download Kinship train/valid/test files if missing.",
    )

    parser.add_argument("--num-qubits", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kappa", type=int, default=1, help="Loss exponent in (y-score)^(2*kappa).")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--val-ratio", type=float, default=0.1, help="Used only for CSV/Toy.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Used only for CSV/Toy.")
    parser.add_argument("--max-triples", type=int, default=0)

    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--eval-max-triples", type=int, default=0, help="0 = use full split.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    parser.add_argument("--output-dir", type=Path, default=Path("runs_fqce"))

    parser.add_argument("--toy-entities", type=int, default=104)
    parser.add_argument("--toy-relations", type=int, default=26)
    parser.add_argument("--toy-triples", type=int, default=3000)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def maybe_limit(triples: list[tuple[int, int, int]], max_triples: int) -> list[tuple[int, int, int]]:
    if max_triples <= 0 or len(triples) <= max_triples:
        return triples
    return triples[:max_triples]


def split_triples(
    triples: list[tuple[int, int, int]], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    idx = list(range(len(triples)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    n = len(idx)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train = [triples[i] for i in idx[:n_train]]
    val = [triples[i] for i in idx[n_train : n_train + n_val]]
    test = [triples[i] for i in idx[n_train + n_val :]]
    return train, val, test


def load_csv_data(path: Path, val_ratio: float, test_ratio: float, seed: int, max_triples: int) -> KGData:
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    entity_to_id: dict[str, int] = {}
    relation_to_id: dict[str, int] = {}
    triples: list[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"entity1", "entity2", "relation"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must include columns {sorted(required)}")

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

    triples = maybe_limit(triples, max_triples)
    train, val, test = split_triples(triples, val_ratio, test_ratio, seed)
    return KGData(train=train, val=val, test=test, entity_to_id=entity_to_id, relation_to_id=relation_to_id)


def parse_kg_line(line: str) -> tuple[str, str, str]:
    line = line.strip()
    if not line:
        raise ValueError("empty line")

    if "\t" in line:
        parts = line.split("\t")
    else:
        parts = line.split()

    if len(parts) < 3:
        raise ValueError(f"invalid triple line: {line}")

    h = parts[0].strip().lower()
    r = parts[1].strip().lower()
    t = parts[2].strip().lower()
    return h, r, t


def ensure_kinship_files(kinship_dir: Path) -> None:
    kinship_dir.mkdir(parents=True, exist_ok=True)
    for split, url in KINSHIP_URLS.items():
        out = kinship_dir / f"{split}.txt"
        if out.exists():
            continue
        urllib.request.urlretrieve(url, out)


def load_kinship_data(kinship_dir: Path, download_if_missing: bool, max_triples: int) -> KGData:
    if download_if_missing:
        ensure_kinship_files(kinship_dir)

    train_path = kinship_dir / "train.txt"
    valid_path = kinship_dir / "valid.txt"
    test_path = kinship_dir / "test.txt"

    for p in (train_path, valid_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing Kinship split file: {p}. "
                f"Use --download-kinship or place train.txt/valid.txt/test.txt in {kinship_dir}."
            )

    entity_to_id: dict[str, int] = {}
    relation_to_id: dict[str, int] = {}

    def encode(h: str, r: str, t: str) -> tuple[int, int, int]:
        if h not in entity_to_id:
            entity_to_id[h] = len(entity_to_id)
        if t not in entity_to_id:
            entity_to_id[t] = len(entity_to_id)
        if r not in relation_to_id:
            relation_to_id[r] = len(relation_to_id)
        return entity_to_id[h], relation_to_id[r], entity_to_id[t]

    def load_split(path: Path) -> list[tuple[int, int, int]]:
        triples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, r, t = parse_kg_line(line)
            triples.append(encode(h, r, t))
        return triples

    train = load_split(train_path)
    val = load_split(valid_path)
    test = load_split(test_path)

    if max_triples > 0:
        train = maybe_limit(train, max_triples)
        val = maybe_limit(val, max(1, max_triples // 10))
        test = maybe_limit(test, max(1, max_triples // 10))

    return KGData(train=train, val=val, test=test, entity_to_id=entity_to_id, relation_to_id=relation_to_id)


def make_toy_data(
    num_entities: int,
    num_relations: int,
    num_triples: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> KGData:
    triples = [
        (
            random.randint(0, num_entities - 1),
            random.randint(0, num_relations - 1),
            random.randint(0, num_entities - 1),
        )
        for _ in range(num_triples)
    ]
    train, val, test = split_triples(triples, val_ratio, test_ratio, seed)
    entity_to_id = {f"e{i}": i for i in range(num_entities)}
    relation_to_id = {f"r{i}": i for i in range(num_relations)}
    return KGData(train=train, val=val, test=test, entity_to_id=entity_to_id, relation_to_id=relation_to_id)


class FQCEContext:
    num_qubits: int = 6
    device = None


def setup_quantum(num_qubits: int) -> None:
    FQCEContext.num_qubits = num_qubits
    FQCEContext.device = qml.device("default.qubit", wires=num_qubits)


def circuit_block(params: torch.Tensor) -> None:
    n = FQCEContext.num_qubits

    for q in range(n):
        qml.Rot(params[0, q, 0], params[0, q, 1], params[0, q, 2], wires=q)

    for block_idx, offset in enumerate((1, 2, 3), start=1):
        for q in range(n):
            target = (q + offset) % n
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
    def sp_circuit(s_params, p_params):
        for q in range(FQCEContext.num_qubits):
            qml.Hadamard(wires=q)
        circuit_block(s_params)
        circuit_block(p_params)
        return qml.state()

    @qml.qnode(FQCEContext.device, interface="torch", diff_method="backprop")
    def o_circuit(o_params):
        for q in range(FQCEContext.num_qubits):
            qml.Hadamard(wires=q)
        circuit_block(o_params)
        return qml.state()

    return sp_circuit, o_circuit


class FQCE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits

        self.entity_params = nn.Parameter(torch.empty(num_entities, 4, num_qubits, 3))
        self.relation_params = nn.Parameter(torch.empty(num_relations, 4, num_qubits, 3))

        init_low = -torch.pi / 10
        init_high = torch.pi / 10
        nn.init.uniform_(self.entity_params, a=float(init_low), b=float(init_high))
        nn.init.uniform_(self.relation_params, a=float(init_low), b=float(init_high))

        self.sp_circuit, self.o_circuit = build_qnodes()

    def score(self, s: int, p: int, o: int) -> torch.Tensor:
        s_params = self.entity_params[s]
        p_params = self.relation_params[p]
        o_params = self.entity_params[o]

        sp_state = self.sp_circuit(s_params, p_params)
        o_state = self.o_circuit(o_params)
        return torch.real(torch.vdot(o_state, sp_state))

    def entity_state(self, e: int) -> torch.Tensor:
        return self.o_circuit(self.entity_params[e])

    def relation_subject_state(self, s: int, p: int) -> torch.Tensor:
        return self.sp_circuit(self.entity_params[s], self.relation_params[p])

    @torch.no_grad()
    def cached_entity_states(self) -> torch.Tensor:
        states = [self.entity_state(e) for e in range(self.entity_params.shape[0])]
        return torch.stack(states, dim=0)


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
        return (s, r, o), self.corrupt(s, r, o)


def mse_label_loss(score: torch.Tensor, y: float, kappa: int) -> torch.Tensor:
    return torch.pow((torch.tensor(y, dtype=score.dtype) - score), 2 * kappa)


@torch.no_grad()
def evaluate_pairwise(
    model: FQCE,
    triples: list[tuple[int, int, int]],
    num_entities: int,
    kappa: int,
    cached_entities: torch.Tensor | None = None,
) -> dict[str, float]:
    if not triples:
        return {"loss": 0.0, "pair_acc": 0.0, "avg_pos": 0.0, "avg_neg": 0.0, "avg_margin": 0.0}

    model.eval()
    dataset = KGDataset(triples, num_entities)
    entity_states = cached_entities if cached_entities is not None else model.cached_entity_states()

    total_loss = 0.0
    correct = 0
    sum_pos = 0.0
    sum_neg = 0.0
    sum_margin = 0.0

    for idx in range(len(dataset)):
        (s, p, o), (sn, pn, on) = dataset[idx]
        pos_sp = model.relation_subject_state(s, p)
        neg_sp = model.relation_subject_state(sn, pn)
        pos = torch.real(torch.vdot(entity_states[o], pos_sp))
        neg = torch.real(torch.vdot(entity_states[on], neg_sp))

        loss = 0.5 * (mse_label_loss(pos, 1.0, kappa) + mse_label_loss(neg, -1.0, kappa))

        total_loss += float(loss.item())
        sum_pos += float(pos.item())
        sum_neg += float(neg.item())
        sum_margin += float((pos - neg).item())
        correct += int(pos.item() > neg.item())

    n = len(dataset)
    return {
        "loss": total_loss / n,
        "pair_acc": correct / n,
        "avg_pos": sum_pos / n,
        "avg_neg": sum_neg / n,
        "avg_margin": sum_margin / n,
    }


@torch.no_grad()
def evaluate_filtered_ranking(
    model: FQCE,
    eval_triples: list[tuple[int, int, int]],
    all_true_triples: set[tuple[int, int, int]],
    num_entities: int,
    max_triples: int,
    cached_entities: torch.Tensor | None = None,
) -> dict[str, float]:
    if not eval_triples:
        return {"mr": 0.0, "mrr": 0.0, "hits@3": 0.0, "hits@10": 0.0, "n": 0.0}

    model.eval()
    triples = eval_triples
    if max_triples > 0 and len(triples) > max_triples:
        triples = random.sample(triples, max_triples)

    entity_states = cached_entities if cached_entities is not None else model.cached_entity_states()
    entity_states_conj = torch.conj(entity_states)
    rel_sub_cache: dict[int, torch.Tensor] = {}

    ranks: list[int] = []

    for s, p, o in triples:
        # Tail ranking
        sp_state = model.relation_subject_state(s, p)
        tail_scores = torch.real((entity_states_conj * sp_state.unsqueeze(0)).sum(dim=1))
        true_tail_score = float(tail_scores[o].item())
        better = 0
        for cand in range(num_entities):
            if cand == o:
                continue
            if (s, p, cand) in all_true_triples:
                continue
            if float(tail_scores[cand].item()) > true_tail_score:
                better += 1
        ranks.append(better + 1)

        # Head ranking
        if p not in rel_sub_cache:
            rel_sub_cache[p] = torch.stack(
                [model.relation_subject_state(cand, p) for cand in range(num_entities)], dim=0
            )
        rel_states = rel_sub_cache[p]
        o_conj = torch.conj(entity_states[o]).unsqueeze(0)
        head_scores = torch.real((rel_states * o_conj).sum(dim=1))
        true_head_score = float(head_scores[s].item())
        better = 0
        for cand in range(num_entities):
            if cand == s:
                continue
            if (cand, p, o) in all_true_triples:
                continue
            cand_score = float(head_scores[cand].item())
            if cand_score > true_head_score:
                better += 1
        ranks.append(better + 1)

    n = len(ranks)
    mr = sum(ranks) / n
    mrr = sum(1.0 / r for r in ranks) / n
    hits3 = sum(1 for r in ranks if r <= 3) / n
    hits10 = sum(1 for r in ranks if r <= 10) / n

    return {"mr": mr, "mrr": mrr, "hits@3": hits3, "hits@10": hits10, "n": float(n)}


def train(args: argparse.Namespace, data: KGData, run_dir: Path) -> None:
    model = FQCE(data.num_entities, data.num_relations, args.num_qubits)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = torch.utils.data.DataLoader(
        KGDataset(data.train, data.num_entities),
        batch_size=args.batch_size,
        shuffle=True,
    )

    all_true = set(data.train) | set(data.val) | set(data.test)

    best_val_hits3 = -1.0
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    history_path = run_dir / "metrics_history.jsonl"
    recent_train_losses: list[float] = []

    LOGGER.info(
        "Training started | epochs=%d batch_size=%d lr=%.4g kappa=%d qubits=%d",
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.kappa,
        args.num_qubits,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        pos_sum = 0.0
        neg_sum = 0.0
        margin_sum = 0.0
        grad_norm_sum = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for pos, neg in loop:
            s, p, o = pos
            sn, pn, on = neg
            s = int(s.item())
            p = int(p.item())
            o = int(o.item())
            sn = int(sn.item())
            pn = int(pn.item())
            on = int(on.item())

            optimizer.zero_grad()

            pos_score = model.score(s, p, o)
            neg_score = model.score(sn, pn, on)

            loss = 0.5 * (
                mse_label_loss(pos_score, 1.0, args.kappa)
                + mse_label_loss(neg_score, -1.0, args.kappa)
            )
            loss.backward()

            grad_norm = 0.0
            if args.grad_clip > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                )

            optimizer.step()

            epoch_loss += float(loss.item())
            total += 1
            pos_val = float(pos_score.item())
            neg_val = float(neg_score.item())
            pos_sum += pos_val
            neg_sum += neg_val
            margin_sum += pos_val - neg_val
            grad_norm_sum += grad_norm
            correct += int(float(pos_score.item()) > float(neg_score.item()))

            loop.set_postfix(train_loss=epoch_loss / max(1, total), pair_acc=correct / max(1, total))

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": epoch_loss / max(1, total),
            "train_pair_acc": correct / max(1, total),
            "train_avg_pos": pos_sum / max(1, total),
            "train_avg_neg": neg_sum / max(1, total),
            "train_avg_margin": margin_sum / max(1, total),
            "train_avg_grad_norm": grad_norm_sum / max(1, total),
            "epoch_seconds": time.time() - epoch_start,
        }
        recent_train_losses.append(row["train_loss"])
        if len(recent_train_losses) > 5:
            recent_train_losses.pop(0)
        row["train_loss_window_slope"] = (
            recent_train_losses[-1] - recent_train_losses[0] if len(recent_train_losses) > 1 else 0.0
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_start = time.time()
            cached_entities = model.cached_entity_states()
            val_pair = evaluate_pairwise(
                model, data.val, data.num_entities, args.kappa, cached_entities=cached_entities
            )
            val_rank = evaluate_filtered_ranking(
                model,
                data.val,
                all_true,
                data.num_entities,
                args.eval_max_triples,
                cached_entities=cached_entities,
            )
            row["eval_seconds"] = time.time() - eval_start

            row.update({f"val_{k}": v for k, v in val_pair.items()})
            row.update({f"val_{k}": v for k, v in val_rank.items()})

            LOGGER.info(
                "Epoch %03d | train_loss=%.4f pair_acc=%.4f margin=%.4f grad=%.4f | "
                "val_hits@3=%.4f val_hits@10=%.4f val_mr=%.2f val_loss=%.4f eval_s=%.1f",
                epoch,
                row["train_loss"],
                row["train_pair_acc"],
                row["train_avg_margin"],
                row["train_avg_grad_norm"],
                val_rank["hits@3"],
                val_rank["hits@10"],
                val_rank["mr"],
                val_pair["loss"],
                row["eval_seconds"],
            )

            if val_rank["hits@3"] > best_val_hits3:
                best_val_hits3 = val_rank["hits@3"]
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
                torch.save(best_state, run_dir / "best_model.pt")
            else:
                stale += 1

            if stale >= args.early_stop_patience:
                LOGGER.info(
                    "Early stopping at epoch %d: no improvement in val Hits@3 for %d eval steps.",
                    epoch,
                    stale,
                )
                history.append(row)
                append_jsonl(history_path, {"timestamp": datetime.now().isoformat(), **row})
                break

        history.append(row)
        append_jsonl(history_path, {"timestamp": datetime.now().isoformat(), **row})

    if best_state is not None:
        model.load_state_dict(best_state)

    cached_entities = model.cached_entity_states()
    test_pair = evaluate_pairwise(
        model, data.test, data.num_entities, args.kappa, cached_entities=cached_entities
    )
    test_rank = evaluate_filtered_ranking(
        model,
        data.test,
        all_true,
        data.num_entities,
        args.eval_max_triples,
        cached_entities=cached_entities,
    )

    LOGGER.info("Best model summary")
    LOGGER.info("best_epoch=%s best_val_hits@3=%.4f", best_epoch, best_val_hits3)
    LOGGER.info(
        "test_loss=%.4f test_pair_acc=%.4f test_mr=%.2f test_hits@3=%.4f test_hits@10=%.4f",
        test_pair["loss"],
        test_pair["pair_acc"],
        test_rank["mr"],
        test_rank["hits@3"],
        test_rank["hits@10"],
    )

    torch.save(model.state_dict(), run_dir / "last_model.pt")

    summary = {
        "best_epoch": best_epoch,
        "best_val_hits@3": best_val_hits3,
        "test_pairwise": test_pair,
        "test_filtered_ranking": test_rank,
        "history": history,
        "n_train": len(data.train),
        "n_val": len(data.val),
        "n_test": len(data.test),
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_quantum(args.num_qubits)

    base_dir = Path(__file__).resolve().parent

    if args.dataset == "kinship":
        kinship_dir = args.kinship_dir
        if not kinship_dir.is_absolute():
            kinship_dir = (base_dir / kinship_dir).resolve()
        data = load_kinship_data(kinship_dir, args.download_kinship, args.max_triples)
    elif args.dataset == "csv":
        csv_path = args.dataset_csv
        if not csv_path.is_absolute():
            csv_path = (base_dir / csv_path).resolve()
        data = load_csv_data(csv_path, args.val_ratio, args.test_ratio, args.seed, args.max_triples)
    else:
        data = make_toy_data(
            args.toy_entities,
            args.toy_relations,
            args.toy_triples,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"fqce_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config = {
        "args": serializable_args,
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
        "n_train": len(data.train),
        "n_val": len(data.val),
        "n_test": len(data.test),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    setup_logging(run_dir, args.log_level)
    LOGGER.info("fQCE training start")
    LOGGER.info(
        "dataset=%s entities=%d relations=%d train=%d val=%d test=%d",
        args.dataset,
        data.num_entities,
        data.num_relations,
        len(data.train),
        len(data.val),
        len(data.test),
    )
    LOGGER.info("run_dir=%s", run_dir)
    if args.dataset == "kinship":
        LOGGER.info("Kinship run is for method validation (benchmark mode).")

    train(args=args, data=data, run_dir=run_dir)


if __name__ == "__main__":
    main()
#This code is designed for training a quantum circuit embedding model (fQCE) for knowledge graph completion, following the VQC-KGE methodology. It includes data loading, model definition, training loop, and evaluation metrics. The code supports the Kinship dataset, custom CSV datasets, and synthetic toy datasets.
# Implemented in training.py:

# Caching optimization (your recommendation)
# Added cached entity-state generation.
# evaluate_pairwise now reuses cached |e_i> instead of recomputing o_circuit() each score.
# evaluate_filtered_ranking now:
# vectorizes tail ranking with cached entity states,
# caches relation-subject states per relation for head ranking.
# This removes a lot of repeated circuit simulation in eval.
# Better logging
# Added structured logger (console + training.log).
# Added --log-level.
# Logs epoch-level train/eval timing and metrics.
# Better training/eval tracking
# Tracks and logs:
# train_loss, train_pair_acc,
# train_avg_pos, train_avg_neg, train_avg_margin,
# train_avg_grad_norm,
# epoch_seconds, eval_seconds,
# train_loss_window_slope (short trend signal).
# Appends per-epoch records to:
# metrics_history.jsonl in run folder.
# Keeps:
# best_model.pt (best val Hits@3),
# last_model.pt.
# Kinship as method-validation mode
# Explicit log note added when running --dataset kinship.
# Validation run:

# python LLM-Simplification/fQCE/training.py --dataset kinship --epochs 1 --eval-every 1 --max-triples 20 --eval-max-triples 5 --num-qubits 6 --log-level INFO
# Completed successfully with new logs/metrics files.
