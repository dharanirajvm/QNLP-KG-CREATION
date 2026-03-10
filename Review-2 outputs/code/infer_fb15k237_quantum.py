import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from training_fb15k237 import QuantumKGE, setup_quantum

# Updated best-model snapshot copy for default inference.
DEFAULT_SNAPSHOT_DIR = (
    Path(__file__).resolve().parent
    / "inference_snapshots"
    / "quantum_fb15k237_20260308_174529_updated_20260310_193344"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference + ranking for saved FB15k-237 quantum model.")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--mode",
        choices=["tail", "head", "score"],
        default="tail",
        help="tail: rank tails for (h,r,?), head: rank heads for (?,r,t), score: score one triple",
    )
    parser.add_argument("--head", type=str, default="")
    parser.add_argument("--relation", type=str, default="")
    parser.add_argument("--tail", type=str, default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--backend",
        type=str,
        default="",
        help="Override backend (e.g. lightning.gpu/default.qubit). Empty => use config backend or default.qubit.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional directory with train.txt/valid.txt/test.txt for filtering known triples.",
    )
    parser.add_argument(
        "--exclude-known",
        action="store_true",
        help="Exclude already-known true triples from ranked candidates (if --dataset-dir is provided).",
    )
    parser.add_argument(
        "--sample-candidates",
        type=int,
        default=0,
        help="If >0, rank only over random sampled candidates (plus target if provided).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=None,
        help="Optional JSON mapping raw FB IDs/relations to readable labels.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive prompt for repeated head/tail/score queries.",
    )
    parser.add_argument("--show-ids", action="store_true", help="Include numeric IDs in output.")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_kg_line(line: str) -> tuple[str, str, str]:
    parts = line.strip().split("\t") if "\t" in line else line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid triple line: {line}")
    return parts[0], parts[1], parts[2]


def build_filter_maps(dataset_dir: Path, entity_to_id: dict[str, int], relation_to_id: dict[str, int]):
    all_true: set[tuple[int, int, int]] = set()
    for split in ("train.txt", "valid.txt", "test.txt"):
        path = dataset_dir / split
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, r, t = parse_kg_line(line)
            if h in entity_to_id and t in entity_to_id and r in relation_to_id:
                all_true.add((entity_to_id[h], relation_to_id[r], entity_to_id[t]))

    tails_filter: dict[tuple[int, int], set[int]] = {}
    heads_filter: dict[tuple[int, int], set[int]] = {}
    for h, r, t in all_true:
        tails_filter.setdefault((h, r), set()).add(t)
        heads_filter.setdefault((r, t), set()).add(h)
    return tails_filter, heads_filter


def resolve_id(text: str, mapping: dict[str, int], name: str) -> int:
    if not text:
        raise ValueError(f"Missing required {name}.")
    if text in mapping:
        return mapping[text]
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(mapping):
            return idx
    lowered = text.lower()
    if lowered in mapping:
        return mapping[lowered]
    raise KeyError(f"Unknown {name}: {text}")


def candidate_ids(total: int, sample_size: int, must_include: int | None, seed: int) -> list[int]:
    if sample_size <= 0 or sample_size >= total:
        return list(range(total))
    rnd = random.Random(seed)
    ids = set(rnd.sample(range(total), sample_size))
    if must_include is not None:
        ids.add(must_include)
    return sorted(ids)


def relation_to_phrase(rel: str) -> str:
    r = rel.strip().strip("/")
    if not r:
        return "related_to"
    toks = [x for x in r.split("/") if x]
    phrase = " ".join(toks)
    return phrase.replace("_", " ")


def display_text(raw: str, labels: dict[str, str]) -> str:
    if raw in labels:
        return labels[raw]
    low = raw.lower()
    if low in labels:
        return labels[low]
    return raw


def triple_sentence(head: str, relation: str, tail: str, labels: dict[str, str]) -> str:
    h = display_text(head, labels)
    r = display_text(relation, labels)
    if r == relation:
        r = relation_to_phrase(r)
    t = display_text(tail, labels)
    return f"{h} -- {r} -- {t}"


def run_query(
    *,
    model: QuantumKGE,
    mode: str,
    head: str,
    relation: str,
    tail: str,
    top_k: int,
    sample_candidates: int,
    seed: int,
    exclude_known: bool,
    tails_filter,
    heads_filter,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    id_to_entity: dict[int, str],
    id_to_relation: dict[int, str],
    labels: dict[str, str],
    show_ids: bool,
) -> dict[str, Any]:
    rel_id = resolve_id(relation, relation_to_id, "relation")
    top_k = max(1, top_k)
    num_entities = len(entity_to_id)

    if mode == "score":
        h_id = resolve_id(head, entity_to_id, "head")
        t_id = resolve_id(tail, entity_to_id, "tail")
        with torch.no_grad():
            score = float(model.score(h_id, rel_id, t_id).item())

        out = {
            "mode": "score",
            "head": display_text(id_to_entity[h_id], labels),
            "relation": display_text(id_to_relation[rel_id], labels),
            "tail": display_text(id_to_entity[t_id], labels),
            "sentence": triple_sentence(id_to_entity[h_id], id_to_relation[rel_id], id_to_entity[t_id], labels),
            "score": score,
        }
        if show_ids:
            out.update({"head_id": h_id, "relation_id": rel_id, "tail_id": t_id})
        return out

    if mode == "tail":
        h_id = resolve_id(head, entity_to_id, "head")
        target_t = resolve_id(tail, entity_to_id, "tail") if tail else None
        cands = candidate_ids(num_entities, sample_candidates, target_t, seed)
        filtered = set()
        if exclude_known and tails_filter is not None:
            filtered = set(tails_filter.get((h_id, rel_id), set()))
            if target_t is not None:
                filtered.discard(target_t)

        with torch.no_grad():
            sp = model.relation_subject_state(h_id, rel_id)
            ent_cache: dict[int, torch.Tensor] = {}
            scored: list[tuple[int, float]] = []
            for c in cands:
                if c in filtered:
                    continue
                if c not in ent_cache:
                    ent_cache[c] = model.entity_state(c)
                s = float(torch.real(torch.vdot(ent_cache[c], sp)).item())
                scored.append((c, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        out = {
            "mode": "tail",
            "head": display_text(id_to_entity[h_id], labels),
            "relation": display_text(id_to_relation[rel_id], labels),
            "top_k": [],
        }
        for i, (c, s) in enumerate(top):
            item = {
                "rank": i + 1,
                "tail": display_text(id_to_entity[c], labels),
                "sentence": triple_sentence(id_to_entity[h_id], id_to_relation[rel_id], id_to_entity[c], labels),
                "score": s,
            }
            if show_ids:
                item["tail_id"] = c
            out["top_k"].append(item)

        if target_t is not None:
            target_score = None
            for c, s in scored:
                if c == target_t:
                    target_score = s
                    break
            if target_score is not None:
                better = sum(1 for c, s in scored if c != target_t and s > target_score)
                target_info = {
                    "tail": display_text(id_to_entity[target_t], labels),
                    "sentence": triple_sentence(id_to_entity[h_id], id_to_relation[rel_id], id_to_entity[target_t], labels),
                    "score": target_score,
                    "rank": better + 1,
                }
                if show_ids:
                    target_info["tail_id"] = target_t
                out["target"] = target_info
        return out

    # head mode
    t_id = resolve_id(tail, entity_to_id, "tail")
    target_h = resolve_id(head, entity_to_id, "head") if head else None
    cands = candidate_ids(num_entities, sample_candidates, target_h, seed)
    filtered = set()
    if exclude_known and heads_filter is not None:
        filtered = set(heads_filter.get((rel_id, t_id), set()))
        if target_h is not None:
            filtered.discard(target_h)

    with torch.no_grad():
        tail_state = model.entity_state(t_id)
        scored: list[tuple[int, float]] = []
        for c in cands:
            if c in filtered:
                continue
            sp_c = model.relation_subject_state(c, rel_id)
            s = float(torch.real(torch.vdot(tail_state, sp_c)).item())
            scored.append((c, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    out = {
        "mode": "head",
        "relation": display_text(id_to_relation[rel_id], labels),
        "tail": display_text(id_to_entity[t_id], labels),
        "top_k": [],
    }
    for i, (c, s) in enumerate(top):
        item = {
            "rank": i + 1,
            "head": display_text(id_to_entity[c], labels),
            "sentence": triple_sentence(id_to_entity[c], id_to_relation[rel_id], id_to_entity[t_id], labels),
            "score": s,
        }
        if show_ids:
            item["head_id"] = c
        out["top_k"].append(item)

    if target_h is not None:
        target_score = None
        for c, s in scored:
            if c == target_h:
                target_score = s
                break
        if target_score is not None:
            better = sum(1 for c, s in scored if c != target_h and s > target_score)
            target_info = {
                "head": display_text(id_to_entity[target_h], labels),
                "sentence": triple_sentence(id_to_entity[target_h], id_to_relation[rel_id], id_to_entity[t_id], labels),
                "score": target_score,
                "rank": better + 1,
            }
            if show_ids:
                target_info["head_id"] = target_h
            out["target"] = target_info
    return out


def interactive_loop(ctx: dict[str, Any], args: argparse.Namespace) -> None:
    print("Interactive mode. Type 'quit' to exit.")
    while True:
        mode = input("mode [tail/head/score]: ").strip().lower()
        if mode in {"quit", "exit", "q"}:
            break
        if mode not in {"tail", "head", "score"}:
            print("Invalid mode.")
            continue

        relation = input("relation: ").strip()
        head = ""
        tail = ""
        if mode in {"tail", "score"}:
            head = input("head: ").strip()
        if mode in {"head", "score"}:
            tail = input("tail: ").strip()

        try:
            out = run_query(
                model=ctx["model"],
                mode=mode,
                head=head,
                relation=relation,
                tail=tail,
                top_k=args.top_k,
                sample_candidates=args.sample_candidates,
                seed=args.seed,
                exclude_known=args.exclude_known,
                tails_filter=ctx["tails_filter"],
                heads_filter=ctx["heads_filter"],
                entity_to_id=ctx["entity_to_id"],
                relation_to_id=ctx["relation_to_id"],
                id_to_entity=ctx["id_to_entity"],
                id_to_relation=ctx["id_to_relation"],
                labels=ctx["labels"],
                show_ids=args.show_ids,
            )
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    snapshot_dir = args.snapshot_dir.resolve()
    entity_to_id = load_json(snapshot_dir / "entity_to_id.json")
    relation_to_id = load_json(snapshot_dir / "relation_to_id.json")
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    labels = {}
    if args.labels_json is not None:
        labels = load_json(args.labels_json.resolve())
    else:
        snap_labels = snapshot_dir / "labels_human.json"
        if snap_labels.exists():
            labels = load_json(snap_labels)
        elif args.dataset_dir is not None:
            ds_labels = args.dataset_dir.resolve() / "labels_human.json"
            if ds_labels.exists():
                labels = load_json(ds_labels)

    cfg = load_json(snapshot_dir / "config.json")
    cfg_args = cfg.get("args", {})
    num_qubits = int(cfg_args.get("num_qubits", 6))
    num_entities = int(cfg.get("num_entities", len(entity_to_id)))
    num_relations = int(cfg.get("num_relations", len(relation_to_id)))
    backend = args.backend or str(cfg_args.get("q_backend", "default.qubit"))

    setup_quantum(num_qubits, backend)
    model = QuantumKGE(num_entities=num_entities, num_relations=num_relations, num_qubits=num_qubits)
    state = torch.load(snapshot_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tails_filter = None
    heads_filter = None
    if args.dataset_dir:
        dataset_dir = args.dataset_dir.resolve()
        tails_filter, heads_filter = build_filter_maps(dataset_dir, entity_to_id, relation_to_id)

    ctx = {
        "model": model,
        "tails_filter": tails_filter,
        "heads_filter": heads_filter,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "id_to_entity": id_to_entity,
        "id_to_relation": id_to_relation,
        "labels": labels,
    }

    if args.interactive:
        interactive_loop(ctx, args)
        return

    if not args.relation:
        raise SystemExit("--relation is required in non-interactive mode.")

    out = run_query(
        model=model,
        mode=args.mode,
        head=args.head,
        relation=args.relation,
        tail=args.tail,
        top_k=args.top_k,
        sample_candidates=args.sample_candidates,
        seed=args.seed,
        exclude_known=args.exclude_known,
        tails_filter=tails_filter,
        heads_filter=heads_filter,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        id_to_entity=id_to_entity,
        id_to_relation=id_to_relation,
        labels=labels,
        show_ids=args.show_ids,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
