import argparse
import json
import logging
import random
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


FB15K237_URLS = {
    "train": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/train.txt",
    "valid": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/valid.txt",
    "test": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/test.txt",
}

LOGGER = logging.getLogger("kge_fb15k237")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Practical end-to-end FB15k-237 KG embedding trainer.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/fb15k237"))
    parser.add_argument("--download", action="store_true")

    parser.add_argument("--model", choices=["complex", "quantum"], default="quantum")
    parser.add_argument(
        "--allow-classical",
        action="store_true",
        help="Allow non-quantum model mode. Keep disabled for strict VQC training.",
    )
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
    parser.add_argument("--q-backend", default="lightning.gpu", help="default.qubit/lightning.qubit/lightning.gpu")
    parser.add_argument("--kappa", type=int, default=1)
    parser.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=0,
        help="If >0, sample this many training triples per epoch (useful for very large KGs).",
    )

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--eval-max-triples", type=int, default=200)
    parser.add_argument("--eval-protocol", choices=["sampled", "exact"], default="sampled")
    parser.add_argument("--eval-candidates", type=int, default=2048)

    parser.add_argument("--max-train-triples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs_kge_fb15k237"))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def setup_logging(run_dir: Path, level: str) -> None:
    LOGGER.setLevel(getattr(logging, level))
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)

    fh = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def parse_kg_line(line: str) -> tuple[str, str, str]:
    parts = line.strip().split("\t") if "\t" in line else line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid triple line: {line}")
    return parts[0], parts[1], parts[2]


def ensure_dataset_files(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, url in FB15K237_URLS.items():
        out = dataset_dir / f"{split}.txt"
        if out.exists():
            continue
        LOGGER.info("Downloading %s to %s", split, out)
        urllib.request.urlretrieve(url, out)


def load_fb15k237(dataset_dir: Path, download: bool, max_train_triples: int) -> KGData:
    if download:
        ensure_dataset_files(dataset_dir)

    train_path = dataset_dir / "train.txt"
    valid_path = dataset_dir / "valid.txt"
    test_path = dataset_dir / "test.txt"
    for p in (train_path, valid_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

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
        triples: list[tuple[int, int, int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, r, t = parse_kg_line(line)
            triples.append(encode(h, r, t))
        return triples

    train = load_split(train_path)
    val = load_split(valid_path)
    test = load_split(test_path)

    if max_train_triples > 0 and len(train) > max_train_triples:
        train = train[:max_train_triples]

    return KGData(train=train, val=val, test=test, entity_to_id=entity_to_id, relation_to_id=relation_to_id)


class ComplexKGE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.ent_re = nn.Embedding(num_entities, dim)
        self.ent_im = nn.Embedding(num_entities, dim)
        self.rel_re = nn.Embedding(num_relations, dim)
        self.rel_im = nn.Embedding(num_relations, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 6.0 / (self.ent_re.embedding_dim ** 0.5)
        nn.init.uniform_(self.ent_re.weight, -bound, bound)
        nn.init.uniform_(self.ent_im.weight, -bound, bound)
        nn.init.uniform_(self.rel_re.weight, -bound, bound)
        nn.init.uniform_(self.rel_im.weight, -bound, bound)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hr = self.ent_re(h)
        hi = self.ent_im(h)
        rr = self.rel_re(r)
        ri = self.rel_im(r)
        tr = self.ent_re(t)
        ti = self.ent_im(t)
        return ((hr * rr - hi * ri) * tr + (hr * ri + hi * rr) * ti).sum(dim=-1)

    @torch.no_grad()
    def score_tails_all(self, h: int, r: int, device: torch.device) -> torch.Tensor:
        h_t = torch.tensor([h], device=device)
        r_t = torch.tensor([r], device=device)
        hr = self.ent_re(h_t)
        hi = self.ent_im(h_t)
        rr = self.rel_re(r_t)
        ri = self.rel_im(r_t)
        q_re = hr * rr - hi * ri
        q_im = hr * ri + hi * rr
        scores = q_re @ self.ent_re.weight.T + q_im @ self.ent_im.weight.T
        return scores.squeeze(0)

    @torch.no_grad()
    def score_heads_all(self, r: int, t: int, device: torch.device) -> torch.Tensor:
        r_t = torch.tensor([r], device=device)
        t_t = torch.tensor([t], device=device)
        rr = self.rel_re(r_t)
        ri = self.rel_im(r_t)
        tr = self.ent_re(t_t)
        ti = self.ent_im(t_t)
        # from ComplEx symmetry for heads
        q_re = rr * tr + ri * ti
        q_im = rr * ti - ri * tr
        scores = self.ent_re.weight @ q_re.squeeze(0) + self.ent_im.weight @ q_im.squeeze(0)
        return scores


class QuantumContext:
    num_qubits: int = 4
    device = None


def setup_quantum(num_qubits: int, backend: str) -> None:
    QuantumContext.num_qubits = num_qubits
    try:
        QuantumContext.device = qml.device(backend, wires=num_qubits)
        LOGGER.info("Using PennyLane backend '%s' with %d qubits", backend, num_qubits)
    except Exception as exc:
        LOGGER.warning("Failed backend '%s' (%s). Falling back to default.qubit", backend, exc)
        QuantumContext.device = qml.device("default.qubit", wires=num_qubits)


def q_circuit_block(params: torch.Tensor) -> None:
    n = QuantumContext.num_qubits
    for q in range(n):
        qml.Rot(params[0, q, 0], params[0, q, 1], params[0, q, 2], wires=q)
    for block_idx, offset in enumerate((1, 2, 3), start=1):
        for q in range(n):
            target = (q + offset) % n
            if target == q:
                continue
            qml.CRot(params[block_idx, q, 0], params[block_idx, q, 1], params[block_idx, q, 2], wires=[q, target])


def build_qnodes():
    @qml.qnode(QuantumContext.device, interface="torch", diff_method="backprop")
    def sp_circuit(s_params, p_params):
        for q in range(QuantumContext.num_qubits):
            qml.Hadamard(wires=q)
        q_circuit_block(s_params)
        q_circuit_block(p_params)
        return qml.state()

    @qml.qnode(QuantumContext.device, interface="torch", diff_method="backprop")
    def e_circuit(e_params):
        for q in range(QuantumContext.num_qubits):
            qml.Hadamard(wires=q)
        q_circuit_block(e_params)
        return qml.state()

    return sp_circuit, e_circuit


class QuantumKGE(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, num_qubits: int):
        super().__init__()
        self.entity_params = nn.Parameter(torch.empty(num_entities, 4, num_qubits, 3))
        self.relation_params = nn.Parameter(torch.empty(num_relations, 4, num_qubits, 3))
        low = -torch.pi / 10
        high = torch.pi / 10
        nn.init.uniform_(self.entity_params, float(low), float(high))
        nn.init.uniform_(self.relation_params, float(low), float(high))
        self.sp_circuit, self.e_circuit = build_qnodes()

    def relation_subject_state(self, h: int, r: int) -> torch.Tensor:
        return self.sp_circuit(self.entity_params[h], self.relation_params[r])

    def entity_state(self, e: int) -> torch.Tensor:
        return self.e_circuit(self.entity_params[e])

    @torch.no_grad()
    def cached_entity_states(self) -> torch.Tensor:
        states = [self.entity_state(e) for e in range(self.entity_params.shape[0])]
        return torch.stack(states, dim=0)

    def score(self, h: int, r: int, t: int) -> torch.Tensor:
        sp = self.relation_subject_state(h, r)
        eo = self.entity_state(t)
        return torch.real(torch.vdot(eo, sp))


class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, triples: list[tuple[int, int, int]]):
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        return self.triples[idx]


def build_filter_maps(all_true: set[tuple[int, int, int]]) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], set[int]]]:
    tails: dict[tuple[int, int], set[int]] = {}
    heads: dict[tuple[int, int], set[int]] = {}
    for h, r, t in all_true:
        tails.setdefault((h, r), set()).add(t)
        heads.setdefault((r, t), set()).add(h)
    return tails, heads


def rank_from_scores(true_idx: int, scores: torch.Tensor, filtered_ids: set[int]) -> int:
    true_score = float(scores[true_idx].item())
    better = 0
    for i in range(scores.shape[0]):
        if i == true_idx:
            continue
        if i in filtered_ids:
            continue
        if float(scores[i].item()) > true_score:
            better += 1
    return better + 1


@torch.no_grad()
def evaluate_filtered_complex(
    model: ComplexKGE,
    triples: list[tuple[int, int, int]],
    tails_filter: dict[tuple[int, int], set[int]],
    heads_filter: dict[tuple[int, int], set[int]],
    num_entities: int,
    device: torch.device,
    protocol: str,
    max_triples: int,
    eval_candidates: int,
) -> dict[str, float]:
    if max_triples > 0 and len(triples) > max_triples:
        triples = random.sample(triples, max_triples)
    if not triples:
        return {"mr": 0.0, "mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0, "n": 0.0}

    ranks: list[int] = []
    for h, r, t in triples:
        tail_scores = model.score_tails_all(h, r, device)
        head_scores = model.score_heads_all(r, t, device)

        if protocol == "sampled":
            # Keep true index + sampled candidates for speed.
            tail_cands = {t}
            head_cands = {h}
            while len(tail_cands) < min(eval_candidates, num_entities):
                tail_cands.add(random.randint(0, num_entities - 1))
            while len(head_cands) < min(eval_candidates, num_entities):
                head_cands.add(random.randint(0, num_entities - 1))

            tail_cands_list = sorted(tail_cands)
            head_cands_list = sorted(head_cands)
            tail_map = {idx: i for i, idx in enumerate(tail_cands_list)}
            head_map = {idx: i for i, idx in enumerate(head_cands_list)}

            tail_sub = tail_scores[torch.tensor(tail_cands_list, device=device)]
            head_sub = head_scores[torch.tensor(head_cands_list, device=device)]

            tail_filtered = {tail_map[x] for x in tails_filter.get((h, r), set()) if x in tail_map and x != t}
            head_filtered = {head_map[x] for x in heads_filter.get((r, t), set()) if x in head_map and x != h}

            tail_rank = rank_from_scores(tail_map[t], tail_sub, tail_filtered)
            head_rank = rank_from_scores(head_map[h], head_sub, head_filtered)
        else:
            tail_filtered = {x for x in tails_filter.get((h, r), set()) if x != t}
            head_filtered = {x for x in heads_filter.get((r, t), set()) if x != h}
            tail_rank = rank_from_scores(t, tail_scores, tail_filtered)
            head_rank = rank_from_scores(h, head_scores, head_filtered)

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


def train_complex(args: argparse.Namespace, data: KGData, run_dir: Path, device: torch.device) -> None:
    model = ComplexKGE(data.num_entities, data.num_relations, args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    dataset = TripleDataset(data.train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    all_true = set(data.train) | set(data.val) | set(data.test)
    tails_filter, heads_filter = build_filter_maps(all_true)

    best_hits3 = -1.0
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    history_path = run_dir / "metrics_history.jsonl"

    recent_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        pair_acc_sum = 0.0
        batches = 0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in loop:
            h = batch[0].to(device)
            r = batch[1].to(device)
            t = batch[2].to(device)
            bsz = h.shape[0]

            neg_t = torch.randint(0, data.num_entities, (bsz, args.negatives_per_positive), device=device)
            h_rep = h.unsqueeze(1).expand(-1, args.negatives_per_positive).reshape(-1)
            r_rep = r.unsqueeze(1).expand(-1, args.negatives_per_positive).reshape(-1)
            neg_flat = neg_t.reshape(-1)

            optimizer.zero_grad()
            pos = model.score(h, r, t)
            neg = model.score(h_rep, r_rep, neg_flat).view(bsz, args.negatives_per_positive)

            loss = F.softplus(-pos).mean() + F.softplus(neg).mean()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            pair_acc = (pos.unsqueeze(1) > neg).float().mean()
            loss_sum += float(loss.item())
            pair_acc_sum += float(pair_acc.item())
            batches += 1
            loop.set_postfix(train_loss=loss_sum / batches, pair_acc=pair_acc_sum / batches)

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(1, batches),
            "train_pair_acc": pair_acc_sum / max(1, batches),
            "epoch_seconds": time.time() - t0,
        }

        recent_losses.append(row["train_loss"])
        if len(recent_losses) > 5:
            recent_losses.pop(0)
        row["train_loss_window_slope"] = recent_losses[-1] - recent_losses[0] if len(recent_losses) > 1 else 0.0

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_t0 = time.time()
            val_metrics = evaluate_filtered_complex(
                model=model,
                triples=data.val,
                tails_filter=tails_filter,
                heads_filter=heads_filter,
                num_entities=data.num_entities,
                device=device,
                protocol=args.eval_protocol,
                max_triples=args.eval_max_triples,
                eval_candidates=args.eval_candidates,
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["eval_seconds"] = time.time() - eval_t0

            LOGGER.info(
                "Epoch %03d | train_loss=%.4f pair_acc=%.4f | val_hits@3=%.4f val_hits@10=%.4f val_mrr=%.4f val_mr=%.2f",
                epoch,
                row["train_loss"],
                row["train_pair_acc"],
                val_metrics["hits@3"],
                val_metrics["hits@10"],
                val_metrics["mrr"],
                val_metrics["mr"],
            )

            if val_metrics["hits@3"] > best_hits3:
                best_hits3 = val_metrics["hits@3"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, run_dir / "best_model.pt")
                stale = 0
            else:
                stale += 1

            if stale >= args.early_stop_patience:
                LOGGER.info("Early stopping at epoch %d after %d non-improving evals.", epoch, stale)
                history.append(row)
                append_jsonl(history_path, {"timestamp": datetime.now().isoformat(), **row})
                break

        history.append(row)
        append_jsonl(history_path, {"timestamp": datetime.now().isoformat(), **row})

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_filtered_complex(
        model=model,
        triples=data.test,
        tails_filter=tails_filter,
        heads_filter=heads_filter,
        num_entities=data.num_entities,
        device=device,
        protocol=args.eval_protocol,
        max_triples=args.eval_max_triples,
        eval_candidates=args.eval_candidates,
    )

    torch.save(model.state_dict(), run_dir / "last_model.pt")
    summary = {
        "model": "complex",
        "best_epoch": best_epoch,
        "best_val_hits@3": best_hits3,
        "test_filtered_ranking": test_metrics,
        "history": history,
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
        "n_train": len(data.train),
        "n_val": len(data.val),
        "n_test": len(data.test),
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info(
        "Test metrics | MR=%.2f MRR=%.4f H@1=%.4f H@3=%.4f H@10=%.4f",
        test_metrics["mr"],
        test_metrics["mrr"],
        test_metrics["hits@1"],
        test_metrics["hits@3"],
        test_metrics["hits@10"],
    )


def train_quantum(args: argparse.Namespace, data: KGData, run_dir: Path) -> None:
    setup_quantum(args.num_qubits, args.q_backend)
    model = QuantumKGE(data.num_entities, data.num_relations, args.num_qubits)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train = data.train
    all_true = set(data.train) | set(data.val) | set(data.test)
    tails_filter, heads_filter = build_filter_maps(all_true)

    best_hits3 = -1.0
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    history_path = run_dir / "metrics_history.jsonl"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.train_samples_per_epoch > 0 and args.train_samples_per_epoch < len(train):
            epoch_train = random.sample(train, args.train_samples_per_epoch)
        else:
            epoch_train = list(train)
        random.shuffle(epoch_train)
        loss_sum = 0.0
        pair_acc = 0

        loop = tqdm(epoch_train, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for h, r, t in loop:
            neg_t = random.randint(0, data.num_entities - 1)
            while neg_t == t:
                neg_t = random.randint(0, data.num_entities - 1)

            optimizer.zero_grad()
            sp = model.relation_subject_state(h, r)
            pos_e = model.entity_state(t)
            neg_e = model.entity_state(neg_t)
            pos = torch.real(torch.vdot(pos_e, sp))
            neg = torch.real(torch.vdot(neg_e, sp))
            loss = 0.5 * ((1.0 - pos) ** (2 * args.kappa) + (-1.0 - neg) ** (2 * args.kappa))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_sum += float(loss.item())
            pair_acc += int(float(pos.item()) > float(neg.item()))
            n = max(1, loop.n + 1)
            loop.set_postfix(train_loss=loss_sum / n, pair_acc=pair_acc / n)

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(1, len(epoch_train)),
            "train_pair_acc": pair_acc / max(1, len(epoch_train)),
            "epoch_seconds": time.time() - t0,
            "train_samples": float(len(epoch_train)),
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # Use sampled protocol by force for quantum practicality.
            val_metrics = evaluate_filtered_quantum(
                model=model,
                triples=data.val,
                tails_filter=tails_filter,
                heads_filter=heads_filter,
                num_entities=data.num_entities,
                max_triples=args.eval_max_triples,
                eval_candidates=args.eval_candidates,
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            LOGGER.info(
                "[Quantum] Epoch %03d | train_loss=%.4f pair_acc=%.4f | val_hits@3=%.4f val_hits@10=%.4f",
                epoch,
                row["train_loss"],
                row["train_pair_acc"],
                val_metrics["hits@3"],
                val_metrics["hits@10"],
            )

            if val_metrics["hits@3"] > best_hits3:
                best_hits3 = val_metrics["hits@3"]
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, run_dir / "best_model.pt")
                stale = 0
            else:
                stale += 1
            if stale >= args.early_stop_patience:
                LOGGER.info("[Quantum] Early stopping at epoch %d", epoch)
                history.append(row)
                append_jsonl(history_path, {"timestamp": datetime.now().isoformat(), **row})
                break

        history.append(row)
        append_jsonl(history_path, {"timestamp": datetime.now().isoformat(), **row})

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_filtered_quantum(
        model=model,
        triples=data.test,
        tails_filter=tails_filter,
        heads_filter=heads_filter,
        num_entities=data.num_entities,
        max_triples=args.eval_max_triples,
        eval_candidates=args.eval_candidates,
    )

    torch.save(model.state_dict(), run_dir / "last_model.pt")
    summary = {
        "model": "quantum",
        "best_epoch": best_epoch,
        "best_val_hits@3": best_hits3,
        "test_filtered_ranking": test_metrics,
        "history": history,
        "num_entities": data.num_entities,
        "num_relations": data.num_relations,
        "n_train": len(data.train),
        "n_val": len(data.val),
        "n_test": len(data.test),
    }
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def evaluate_filtered_quantum(
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
        tail_scores = {}
        for c in tail_cands:
            tail_scores[c] = float(torch.real(torch.vdot(cached_entities[c], sp_state)).item())

        o_state_conj = torch.conj(cached_entities[t])
        head_scores = {}
        for c in head_cands:
            sp_c = model.relation_subject_state(c, r)
            head_scores[c] = float(torch.real(torch.vdot(o_state_conj, sp_c)).item())

        tail_true = tail_scores[t]
        tail_filtered = tails_filter.get((h, r), set())
        tail_rank = 1 + sum(
            1 for c, s in tail_scores.items() if c != t and c not in tail_filtered and s > tail_true
        )

        head_true = head_scores[h]
        head_filtered = heads_filter.get((r, t), set())
        head_rank = 1 + sum(
            1 for c, s in head_scores.items() if c != h and c not in head_filtered and s > head_true
        )

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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.model == "complex" and not args.allow_classical:
        raise SystemExit(
            "Classical mode is disabled for strict VQC training. "
            "Use --model quantum (default), or pass --allow-classical explicitly."
        )

    base_dir = Path(__file__).resolve().parent
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else (base_dir / args.dataset_dir).resolve()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.model}_fb15k237_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir, args.log_level)

    data = load_fb15k237(dataset_dir=dataset_dir, download=args.download, max_train_triples=args.max_train_triples)

    device = resolve_device(args.device)
    LOGGER.info("Training start | model=%s device=%s", args.model, device)
    if args.model == "quantum":
        LOGGER.info("Quantum backend requested: %s", args.q_backend)
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
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "entity_to_id.json").write_text(json.dumps(data.entity_to_id, indent=2), encoding="utf-8")
    (run_dir / "relation_to_id.json").write_text(json.dumps(data.relation_to_id, indent=2), encoding="utf-8")

    if args.model == "complex":
        train_complex(args=args, data=data, run_dir=run_dir, device=device)
    else:
        train_quantum(args=args, data=data, run_dir=run_dir)


if __name__ == "__main__":
    main()
