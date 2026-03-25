#!/usr/bin/env python3
"""Shared utilities for downstream tasks on module1 KGE runs."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
FQCE_DIR = THIS_DIR.parent
PROJECT_ROOT = FQCE_DIR.parent.parent
if str(FQCE_DIR) not in sys.path:
    sys.path.insert(0, str(FQCE_DIR))

from training_fb15k237 import ComplexKGE, QuantumKGE, setup_quantum


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_kg_line(line: str) -> tuple[str, str, str]:
    parts = line.strip().split("\t") if "\t" in line else line.strip().split()
    if len(parts) < 3:
        raise ValueError(f"Invalid triple line: {line}")
    return parts[0], parts[1], parts[2]


def norm_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def humanize(raw: str) -> str:
    return raw.strip("/").replace("_", " ").replace("/", " ")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def relation_aliases(raw_relation: str) -> set[str]:
    base = humanize(raw_relation)
    aliases = {raw_relation.lower(), base.lower()}
    if raw_relation == "studies_at":
        aliases.update({"study", "studied", "studies", "student", "students", "university", "college", "school"})
    elif raw_relation == "works_at":
        aliases.update({"work", "works", "worked", "job", "company", "employer", "employee"})
    elif raw_relation == "lives_in":
        aliases.update({"live", "lives", "living", "reside", "resides", "city", "country", "place"})
    elif raw_relation == "born_in":
        aliases.update({"born", "birthplace", "birth", "from"})
    elif raw_relation == "located_in":
        aliases.update({"located", "location", "in", "at"})
    elif raw_relation == "part_of":
        aliases.update({"part", "component", "piece", "inside", "within"})
    elif raw_relation == "married_to":
        aliases.update({"married", "spouse", "wife", "husband", "partner"})
    elif raw_relation == "parent_of":
        aliases.update({"parent", "child", "children", "father", "mother", "son", "daughter"})
    elif raw_relation == "owns":
        aliases.update({"own", "owns", "owned", "has", "have", "possess"})
    elif raw_relation == "uses":
        aliases.update({"use", "uses", "used", "tool", "software", "platform", "technology"})

    rel_lower = raw_relation.lower()
    base_lower = base.lower()
    if "place_of_birth" in rel_lower or "place of birth" in base_lower:
        aliases.update({"born", "birth", "birthplace", "where born"})
    if "nationality" in rel_lower or "nationality" in base_lower:
        aliases.update({"nationality", "citizen", "country", "from"})
    if "profession" in rel_lower or "profession" in base_lower:
        aliases.update({"profession", "job", "occupation", "work as", "career"})
    if "education" in rel_lower or "institution" in rel_lower or "school" in rel_lower:
        aliases.update({"study", "studied", "education", "educated", "school", "college", "university"})
    if "place_lived" in rel_lower or "residence" in rel_lower or "location" in rel_lower:
        aliases.update({"live", "lives", "living", "reside", "location", "where"})
    if "country" in rel_lower:
        aliases.update({"country", "nation"})
    if "capital" in rel_lower:
        aliases.update({"capital", "capital city"})
    if "spouse" in rel_lower or "marriage" in rel_lower:
        aliases.update({"spouse", "married", "wife", "husband", "partner"})
    return aliases


@dataclass
class KGEContext:
    snapshot_dir: Path
    dataset_dir: Path
    model_name: str
    model: object
    entity_to_id: dict[str, int]
    relation_to_id: dict[str, int]
    id_to_entity: dict[int, str]
    id_to_relation: dict[int, str]
    labels: dict[str, str]
    true_triples: set[tuple[int, int, int]]
    tails_filter: dict[tuple[int, int], set[int]]
    heads_filter: dict[tuple[int, int], set[int]]
    embedding_cache: dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)

    def display(self, raw: str) -> str:
        return self.labels.get(raw, self.labels.get(raw.lower(), raw))

    def resolve_entity(self, text: str) -> int:
        return self._resolve_text_id(text=text, mapping=self.entity_to_id, labels_scope="entity")

    def resolve_relation(self, text: str) -> int:
        return self._resolve_text_id(text=text, mapping=self.relation_to_id, labels_scope="relation")

    def _resolve_text_id(self, text: str, mapping: dict[str, int], labels_scope: str) -> int:
        s = norm_text(text)
        if not s:
            raise ValueError(f"Missing {labels_scope}")
        if s in mapping:
            return mapping[s]
        if s.isdigit():
            idx = int(s)
            if 0 <= idx < len(mapping):
                return idx

        for raw, idx in mapping.items():
            label = norm_text(self.display(raw))
            if s == label:
                return idx

        exact_token_matches = []
        substring_matches = []
        s_tokens = set(tokenize(s))
        for raw, idx in mapping.items():
            label = norm_text(self.display(raw))
            label_tokens = set(tokenize(label))
            raw_tokens = set(tokenize(raw))
            if s_tokens and s_tokens == label_tokens:
                exact_token_matches.append(idx)
            elif s in label or s in raw:
                substring_matches.append(idx)
            elif s_tokens and s_tokens.issubset(label_tokens | raw_tokens):
                substring_matches.append(idx)

        matches = exact_token_matches or substring_matches
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"Unknown {labels_scope}: {text}")
        raise KeyError(f"Ambiguous {labels_scope}: {text}")

    def entity_embedding(self, entity_id: int) -> torch.Tensor:
        cached = self.embedding_cache.get(entity_id)
        if cached is not None:
            return cached
        if self.model_name == "complex":
            vec = torch.cat(
                [self.model.ent_re.weight[entity_id], self.model.ent_im.weight[entity_id]],
                dim=0,
            ).float()
        else:
            state = self.model.entity_state(entity_id)
            vec = torch.cat([state.real, state.imag], dim=0).float()
        vec = F.normalize(vec.unsqueeze(0), dim=1).squeeze(0)
        self.embedding_cache[entity_id] = vec
        return vec

    def similarity(self, entity_a_id: int, entity_b_id: int) -> float:
        emb_a = self.entity_embedding(entity_a_id)
        emb_b = self.entity_embedding(entity_b_id)
        return float(torch.dot(emb_a, emb_b).item())

    def rank_similar_entities(
        self,
        anchor_id: int,
        *,
        top_k: int = 10,
        exclude_self: bool = True,
        candidate_ids: list[int] | None = None,
    ) -> list[dict]:
        candidate_list = list(candidate_ids) if candidate_ids is not None else list(range(self.num_entities))
        if exclude_self:
            candidate_list = [entity_id for entity_id in candidate_list if entity_id != anchor_id]
        if not candidate_list:
            return []

        anchor_vec = self.entity_embedding(anchor_id)
        emb_matrix = torch.stack([self.entity_embedding(entity_id) for entity_id in candidate_list], dim=0)
        scores = torch.mv(emb_matrix, anchor_vec)
        top_n = min(top_k, len(candidate_list))
        vals, idxs = torch.topk(scores, k=top_n)

        rows = []
        for rank, (score, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
            neighbor_id = candidate_list[idx]
            rows.append(
                {
                    "rank": rank,
                    "anchor_id": anchor_id,
                    "anchor_raw": self.id_to_entity[anchor_id],
                    "anchor": self.display(self.id_to_entity[anchor_id]),
                    "neighbor_id": neighbor_id,
                    "neighbor_raw": self.id_to_entity[neighbor_id],
                    "neighbor": self.display(self.id_to_entity[neighbor_id]),
                    "cosine_similarity": float(score),
                }
            )
        return rows

    def score(self, head_id: int, relation_id: int, tail_id: int) -> float:
        with torch.no_grad():
            if self.model_name == "complex":
                h = torch.tensor([head_id], dtype=torch.long)
                r = torch.tensor([relation_id], dtype=torch.long)
                t = torch.tensor([tail_id], dtype=torch.long)
                return float(self.model.score(h, r, t).item())
            return float(self.model.score(head_id, relation_id, tail_id).item())

    def sentence_for_ids(self, head_id: int, relation_id: int, tail_id: int) -> str:
        return f"{self.display(self.id_to_entity[head_id])} -- {self.display(self.id_to_relation[relation_id])} -- {self.display(self.id_to_entity[tail_id])}"

    def rank_tails(
        self,
        head_id: int,
        relation_id: int,
        *,
        top_k: int = 10,
        exclude_known: bool = True,
        allow_self: bool = False,
        candidate_ids: list[int] | None = None,
    ) -> list[dict]:
        filtered = set(self.tails_filter.get((head_id, relation_id), set())) if exclude_known else set()
        rows = []
        candidate_iter = candidate_ids if candidate_ids is not None else range(self.num_entities)
        for tail_id in candidate_iter:
            if not allow_self and tail_id == head_id:
                continue
            if tail_id in filtered:
                continue
            score = self.score(head_id, relation_id, tail_id)
            rows.append(
                {
                    "head_id": head_id,
                    "relation_id": relation_id,
                    "tail_id": tail_id,
                    "head": self.display(self.id_to_entity[head_id]),
                    "relation": self.display(self.id_to_relation[relation_id]),
                    "tail": self.display(self.id_to_entity[tail_id]),
                    "raw_tail": self.id_to_entity[tail_id],
                    "score": score,
                    "sentence": self.sentence_for_ids(head_id, relation_id, tail_id),
                }
            )
        rows.sort(key=lambda x: x["score"], reverse=True)
        for idx, row in enumerate(rows[:top_k], start=1):
            row["rank"] = idx
        return rows[:top_k]

    def rank_heads(
        self,
        relation_id: int,
        tail_id: int,
        *,
        top_k: int = 10,
        exclude_known: bool = True,
        allow_self: bool = False,
        candidate_ids: list[int] | None = None,
    ) -> list[dict]:
        filtered = set(self.heads_filter.get((relation_id, tail_id), set())) if exclude_known else set()
        rows = []
        candidate_iter = candidate_ids if candidate_ids is not None else range(self.num_entities)
        for head_id in candidate_iter:
            if not allow_self and head_id == tail_id:
                continue
            if head_id in filtered:
                continue
            score = self.score(head_id, relation_id, tail_id)
            rows.append(
                {
                    "head_id": head_id,
                    "relation_id": relation_id,
                    "tail_id": tail_id,
                    "head": self.display(self.id_to_entity[head_id]),
                    "relation": self.display(self.id_to_relation[relation_id]),
                    "tail": self.display(self.id_to_entity[tail_id]),
                    "raw_head": self.id_to_entity[head_id],
                    "score": score,
                    "sentence": self.sentence_for_ids(head_id, relation_id, tail_id),
                }
            )
        rows.sort(key=lambda x: x["score"], reverse=True)
        for idx, row in enumerate(rows[:top_k], start=1):
            row["rank"] = idx
        return rows[:top_k]

    def rank_relations(
        self,
        head_id: int,
        tail_id: int,
        *,
        top_k: int = 10,
        candidate_ids: list[int] | None = None,
    ) -> list[dict]:
        rows = []
        candidate_iter = candidate_ids if candidate_ids is not None else range(self.num_relations)
        for relation_id in candidate_iter:
            score = self.score(head_id, relation_id, tail_id)
            rows.append(
                {
                    "head_id": head_id,
                    "relation_id": relation_id,
                    "tail_id": tail_id,
                    "head": self.display(self.id_to_entity[head_id]),
                    "relation": self.display(self.id_to_relation[relation_id]),
                    "tail": self.display(self.id_to_entity[tail_id]),
                    "raw_relation": self.id_to_relation[relation_id],
                    "score": score,
                    "sentence": self.sentence_for_ids(head_id, relation_id, tail_id),
                }
            )
        rows.sort(key=lambda x: x["score"], reverse=True)
        for idx, row in enumerate(rows[:top_k], start=1):
            row["rank"] = idx
        return rows[:top_k]

    def known_answers(self, *, head_id: int | None = None, relation_id: int | None = None, tail_id: int | None = None) -> list[tuple[int, int, int]]:
        out = []
        for h, r, t in self.true_triples:
            if head_id is not None and h != head_id:
                continue
            if relation_id is not None and r != relation_id:
                continue
            if tail_id is not None and t != tail_id:
                continue
            out.append((h, r, t))
        return out

    def entity_document(self, entity_id: int) -> str:
        raw = self.id_to_entity[entity_id]
        pieces = [raw, self.display(raw)]
        rel_neighbors = []
        for h, r, t in self.true_triples:
            if h == entity_id:
                rel_neighbors.append(f"{humanize(self.id_to_relation[r])} {self.display(self.id_to_entity[t])}")
            if t == entity_id:
                rel_neighbors.append(f"{humanize(self.id_to_relation[r])} {self.display(self.id_to_entity[h])}")
        pieces.extend(rel_neighbors[:20])
        return " ".join(norm_text(x) for x in pieces if x)


def infer_dataset_dir(snapshot_dir: Path, dataset_dir: Path | None) -> Path:
    if dataset_dir is not None:
        return dataset_dir.resolve()
    summary_path = snapshot_dir / "pipeline_summary_module1.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        ds = summary.get("artifacts", {}).get("dataset_dir")
        if ds:
            return Path(ds).resolve()

    config_path = snapshot_dir / "config.json"
    if config_path.exists():
        config = load_json(config_path)
        ds_cfg = config.get("args", {}).get("dataset_dir")
        if ds_cfg:
            ds_path = Path(str(ds_cfg))
            candidates = []
            if ds_path.is_absolute():
                candidates.append(ds_path)
            else:
                candidates.append((snapshot_dir.parent / ds_path).resolve())
                candidates.append((FQCE_DIR / ds_path).resolve())
                candidates.append((PROJECT_ROOT / ds_path).resolve())
            for candidate in candidates:
                if (candidate / "train.txt").exists():
                    return candidate.resolve()

    raise FileNotFoundError(
        "Dataset directory not provided and could not be inferred from pipeline summary or config args.dataset_dir."
    )


def build_truth_maps(dataset_dir: Path, entity_to_id: dict[str, int], relation_to_id: dict[str, int]):
    all_true: set[tuple[int, int, int]] = set()
    for split in ("train.txt", "valid.txt", "test.txt"):
        path = dataset_dir / split
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, r, t = parse_kg_line(line)
            if h in entity_to_id and r in relation_to_id and t in entity_to_id:
                all_true.add((entity_to_id[h], relation_to_id[r], entity_to_id[t]))

    tails_filter: dict[tuple[int, int], set[int]] = {}
    heads_filter: dict[tuple[int, int], set[int]] = {}
    for h, r, t in all_true:
        tails_filter.setdefault((h, r), set()).add(t)
        heads_filter.setdefault((r, t), set()).add(h)
    return all_true, tails_filter, heads_filter


def load_model(snapshot_dir: Path):
    config = load_json(snapshot_dir / "config.json")
    entity_to_id = load_json(snapshot_dir / "entity_to_id.json")
    relation_to_id = load_json(snapshot_dir / "relation_to_id.json")
    labels_path = snapshot_dir / "labels_human.json"
    labels = load_json(labels_path) if labels_path.exists() else {}

    cfg_args = config.get("args", {})
    num_entities = int(config.get("num_entities", len(entity_to_id)))
    num_relations = int(config.get("num_relations", len(relation_to_id)))
    model_name = str(cfg_args.get("model", "quantum"))
    model_path = snapshot_dir / "best_model.pt"
    if not model_path.exists():
        model_path = snapshot_dir / "last_model.pt"

    if model_name == "complex":
        dim = int(cfg_args.get("embedding_dim", 256))
        model = ComplexKGE(num_entities=num_entities, num_relations=num_relations, dim=dim)
    else:
        num_qubits = int(cfg_args.get("num_qubits", 6))
        backend = str(cfg_args.get("q_backend", "default.qubit"))
        setup_quantum(num_qubits, backend)
        model = QuantumKGE(num_entities=num_entities, num_relations=num_relations, num_qubits=num_qubits)
        model_name = "quantum"

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model_name, model, entity_to_id, relation_to_id, labels


def load_context(snapshot_dir: Path, dataset_dir: Path | None = None) -> KGEContext:
    snapshot_dir = snapshot_dir.resolve()
    dataset_dir = infer_dataset_dir(snapshot_dir, dataset_dir)
    model_name, model, entity_to_id, relation_to_id, labels = load_model(snapshot_dir)
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    true_triples, tails_filter, heads_filter = build_truth_maps(dataset_dir, entity_to_id, relation_to_id)
    return KGEContext(
        snapshot_dir=snapshot_dir,
        dataset_dir=dataset_dir,
        model_name=model_name,
        model=model,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        id_to_entity=id_to_entity,
        id_to_relation=id_to_relation,
        labels=labels,
        true_triples=true_triples,
        tails_filter=tails_filter,
        heads_filter=heads_filter,
    )


def detect_relation_from_query(query: str, relation_to_id: dict[str, int]) -> str | None:
    q_tokens = set(tokenize(query))
    best_rel = None
    best_score = 0
    for rel in relation_to_id.keys():
        aliases = relation_aliases(rel)
        score = len(q_tokens & set(tokenize(" ".join(sorted(aliases)))))
        if score > best_score:
            best_rel = rel
            best_score = score
    return best_rel if best_score > 0 else None


def detect_entities_in_query(query: str, ctx: KGEContext, limit: int = 3) -> list[int]:
    q = norm_text(query)
    matches = []
    for entity_id, raw in ctx.id_to_entity.items():
        label = norm_text(ctx.display(raw))
        if raw in q or label in q:
            matches.append((len(label), entity_id))
    matches.sort(reverse=True)
    return [entity_id for _, entity_id in matches[:limit]]
