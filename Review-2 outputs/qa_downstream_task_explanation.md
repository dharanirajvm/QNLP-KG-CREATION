# Question Answering (QA) Downstream Task: Detailed Implementation Guide

## Overview

The QA downstream task implements a **hybrid knowledge-based question answering system** that combines:
- **Exact graph lookup** for known facts from the training triples
- **Knowledge Graph Embedding (KGE) completion** for predicting missing answers
- **Pattern-based question parsing** to convert natural language questions into structured KG queries

This approach enables precise factual question answering by leveraging both explicit knowledge from the training graph and learned relational patterns from KGE models.

## Detailed Implementation Steps

### Phase 1: Question Parsing and Pattern Matching

#### 1.1 Question Pattern Recognition

**Purpose:** Convert natural language questions into structured (entity, relation, direction) queries.

**Pattern-Based Parsing:**
```python
QUESTION_PATTERNS = [
    (re.compile(r"where\s+was\s+(.+?)\s+born\??$", re.I), "/people/person/place_of_birth", "tail"),
    (re.compile(r"what\s+is\s+(.+?)'s\s+profession\??$", re.I), "/people/person/profession", "tail"),
    (re.compile(r"what\s+is\s+the\s+profession\s+of\s+(.+?)\??$", re.I), "/people/person/profession", "tail"),
    (re.compile(r"what\s+is\s+(.+?)'s\s+nationality\??$", re.I), "/people/person/nationality", "tail"),
    (re.compile(r"which\s+country\s+is\s+(.+?)\s+from\??$", re.I), "/people/person/nationality", "tail"),
    (re.compile(r"where\s+did\s+(.+?)\s+study\??$", re.I), "/education/educational_degree/people_with_this_degree./education/education/institution", "tail"),
]
```

**Parsing Algorithm:**
```python
def parse_question(question: str) -> dict:
    for pattern, relation_raw, direction in QUESTION_PATTERNS:
        match = pattern.match(question.strip())
        if match:
            return {
                "entity_text": match.group(1).strip(),  # Extracted entity name
                "relation_raw": relation_raw,           # Mapped relation
                "direction": direction                  # Query direction
            }
    return {"entity_text": None, "relation_raw": None, "direction": None}
```

**Pattern Examples:**
- **"Where was Barack Obama born?"** → `{"entity_text": "Barack Obama", "relation_raw": "/people/person/place_of_birth", "direction": "tail"}`
- **"What is Marie Curie's profession?"** → `{"entity_text": "Marie Curie", "relation_raw": "/people/person/profession", "direction": "tail"}`

#### 1.2 Fallback Entity/Relation Detection

**Purpose:** Handle questions that don't match predefined patterns using NLP-based detection.

**Algorithm:**
```python
def answer_question_fb15k(ctx, question: str, top_k: int) -> dict:
    parsed = parse_question(question)
    if parsed["entity_text"] is None or parsed["relation_raw"] is None:
        # Fallback: Use general entity/relation detection
        entity_hits = detect_entities_in_query(question, ctx, limit=1)
        relation_raw = detect_relation_from_query(question, ctx.relation_to_id)
        if not entity_hits or relation_raw is None:
            raise ValueError("Could not parse question into a FB15k KG query.")
        parsed = {
            "entity_text": ctx.display(ctx.id_to_entity[entity_hits[0]]),
            "relation_raw": relation_raw,
            "direction": "tail",
        }
    # ... continue with resolution
```

### Phase 2: Entity and Relation Resolution

#### 2.1 Entity ID Resolution

**Purpose:** Map extracted entity names to internal KG entity IDs.

**Algorithm:**
```python
entity_id = ctx.resolve_entity(parsed["entity_text"])
```

**Resolution Process:**
1. **Direct mapping:** Check if entity_text exists in `entity_to_id`
2. **Display name matching:** Check against humanized entity names
3. **Substring matching:** Find entities containing the query text
4. **Ranking:** Prefer longer, more specific matches

#### 2.2 Relation ID Resolution

**Purpose:** Map relation names to internal KG relation IDs.

**Algorithm:**
```python
relation_id = ctx.resolve_relation(parsed["relation_raw"])
```

**Similar resolution process as entities, with fallback to alias matching.**

### Phase 3: Knowledge Graph Lookup (Exact Facts)

#### 3.1 Known Answers Retrieval

**Purpose:** Extract exact facts from the training knowledge graph.

**Algorithm:**
```python
def known_answers(self, *, head_id: int | None = None, relation_id: int | None = None, tail_id: int | None = None) -> list[tuple[int, int, int]]:
    out = []
    for h, r, t in self.true_triples:  # Training triples
        if head_id is not None and h != head_id:
            continue
        if relation_id is not None and r != relation_id:
            continue
        if tail_id is not None and t != tail_id:
            continue
        out.append((h, r, t))
    return out
```

**Query Construction:**
- **Tail prediction:** Find all `(head, relation, ?)` where head and relation are known
- **Head prediction:** Find all `(?, relation, tail)` where relation and tail are known

**Example for "Where was Obama born?":**
- Query: `(Obama, place_of_birth, ?)`
- Known answers: `[(Obama, place_of_birth, Honolulu)]` (if exists in training data)

### Phase 4: KGE-Based Answer Prediction

#### 4.1 KGE Scoring for All Candidates

**Purpose:** Use trained embeddings to score all possible answer entities.

**Tail Prediction Algorithm:**
```python
def rank_tails(self, head_id: int, relation_id: int, *, top_k: int = 10, exclude_known: bool = True) -> list[dict]:
    filtered = set(self.tails_filter.get((head_id, relation_id), set())) if exclude_known else set()
    rows = []

    for tail_id in range(self.num_entities):  # Score ALL entities
        if tail_id in filtered:
            continue
        score = self.score(head_id, relation_id, tail_id)  # KGE scoring

        rows.append({
            "tail_id": tail_id,
            "tail": self.display(self.id_to_entity[tail_id]),
            "score": score,
            "sentence": self.sentence_for_ids(head_id, relation_id, tail_id),
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_k]
```

**Head Prediction Algorithm:**
```python
def rank_heads(self, relation_id: int, tail_id: int, *, top_k: int = 10, exclude_known: bool = True) -> list[dict]:
    # Similar logic but ranking heads instead of tails
    filtered = set(self.heads_filter.get((relation_id, tail_id), set())) if exclude_known else set()
    rows = []

    for head_id in range(self.num_entities):
        if head_id in filtered:
            continue
        score = self.score(head_id, relation_id, tail_id)

        rows.append({
            "head_id": head_id,
            "head": self.display(self.id_to_entity[head_id]),
            "score": score,
            "sentence": self.sentence_for_ids(head_id, relation_id, tail_id),
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_k]
```

#### 4.2 KGE Scoring Implementation

**For ComplEx Model:**
```python
def score(self, h: int, r: int, t: int) -> float:
    # Extract real and imaginary parts
    h_real, h_imag = self.entity_emb(h)[:self.embedding_dim], self.entity_emb(h)[self.embedding_dim:]
    r_real, r_imag = self.relation_emb(r)[:self.embedding_dim], self.relation_emb(r)[self.embedding_dim:]
    t_real, t_imag = self.entity_emb(t)[:self.embedding_dim], self.entity_emb(t)[self.embedding_dim:]

    # ComplEx scoring: Re(<head, relation, conj(tail)>)
    real_part = (h_real * r_real * t_real + h_imag * r_imag * t_real +
                h_real * r_imag * t_imag - h_imag * r_real * t_imag)
    return float(real_part.sum().item())
```

**For Quantum Model:**
```python
def score(self, h: int, r: int, t: int) -> float:
    sp = self.relation_subject_state(h, r)  # Encode (head, relation)
    eo = self.entity_state(t)               # Encode tail
    return float(torch.real(torch.vdot(eo, sp)).item())  # Quantum inner product
```

### Phase 5: Answer Merging and Ranking

#### 5.1 Combining Known and Predicted Answers

**Algorithm:**
```python
# Get known answers from training graph
known_answers = [
    {
        "answer": ctx.display(ctx.id_to_entity[t]),
        "sentence": ctx.sentence_for_ids(entity_id, relation_id, t),
        "source": "known_kg"
    }
    for _, _, t in ctx.known_answers(head_id=entity_id, relation_id=relation_id)
]

# Get KGE predictions
predicted_answers = [
    {
        "answer": row["tail"],
        "sentence": row["sentence"],
        "score": row["score"],
        "source": "kge_completion",
    }
    for row in ranked
]

# Merge and deduplicate
merged = []
seen = set()
for item in known_answers + predicted_answers:
    key = item["answer"].lower()
    if key in seen:
        continue
    seen.add(key)
    merged.append(item)

# Return top-k answers
return merged[:top_k]
```

**Key Features:**
- **Deduplication:** Remove duplicate answers (case-insensitive)
- **Priority:** Known answers appear first, then KGE predictions
- **Scoring:** KGE predictions include confidence scores

## Complete Example: "Where was Barack Obama born?"

### Step-by-Step Execution:

1. **Question Parsing:**
   - Pattern match: `r"where\s+was\s+(.+?)\s+born\??$"`
   - Extracted entity: "Barack Obama"
   - Mapped relation: "/people/person/place_of_birth"
   - Direction: "tail"

2. **Entity Resolution:**
   - "Barack Obama" → entity_id = 1234 (example)
   - "/people/person/place_of_birth" → relation_id = 56

3. **Known Answers Lookup:**
   - Query training triples: `(Obama, place_of_birth, ?)`
   - Found: `(Obama, place_of_birth, Honolulu)` ✓
   - Answer: "Honolulu" (source: "known_kg")

4. **KGE Prediction (if needed):**
   - Score all entities: `score(Obama, place_of_birth, candidate)`
   - Top predictions: Honolulu (0.95), Chicago (0.23), Hawaii (0.18), etc.
   - Used when no known answer exists

5. **Final Answer Ranking:**
   - Known answers prioritized
   - KGE predictions fill gaps
   - Return top-k with sources and scores

## Technical Implementation Details

### Answer Structure:
```json
{
  "task": "qa_answering_fb15k237",
  "question": "Where was Barack Obama born?",
  "parsed_query": {
    "head": "Barack Obama",
    "relation": "place of birth",
    "tail": "?"
  },
  "answers": [
    {
      "answer": "Honolulu",
      "sentence": "Barack Obama -- place of birth -- Honolulu",
      "source": "known_kg"
    },
    {
      "answer": "Chicago",
      "sentence": "Barack Obama -- place of birth -- Chicago",
      "score": 0.234567,
      "source": "kge_completion"
    }
  ]
}
```

### Filtering Strategy:
- **exclude_known=True:** Prevents KGE from predicting already known facts
- **filtered sets:** Pre-computed sets of entities that should be excluded (from validation/test sets)

### Performance Characteristics:

**Strengths:**
1. **Factual Accuracy:** Known answers from training data are guaranteed correct
2. **Coverage:** KGE predictions handle incomplete knowledge graphs
3. **Efficiency:** Exact lookup for known facts, selective KGE scoring
4. **Explainability:** Clear distinction between factual and predicted answers

**Limitations:**
1. **Pattern Dependency:** Limited to predefined question patterns
2. **Entity Resolution:** Requires exact or near-exact entity name matching
3. **Single-Hop Only:** Cannot handle multi-hop reasoning questions

## Advanced Features

### Answer Validation:
- **Source Attribution:** Each answer tagged with "known_kg" or "kge_completion"
- **Confidence Scores:** KGE predictions include probability-like scores
- **Sentence Generation:** Human-readable fact statements

### Scalability:
- **Batch Processing:** Could be extended to batch multiple questions
- **Index Optimization:** Entity/relation resolution could use inverted indexes
- **GPU Acceleration:** KGE scoring benefits from GPU parallelization

## Integration with KGE Training

The QA task directly leverages the trained KGE model to:
1. **Complete missing facts** in the knowledge graph
2. **Provide confidence scores** for predicted answers
3. **Handle out-of-distribution queries** not seen in training
4. **Enable zero-shot QA** for relations learned during training

This creates a powerful hybrid system where explicit knowledge (training triples) and learned patterns (KGE embeddings) work together to answer factual questions with high precision and recall.</content>
<parameter name="filePath">c:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-2 outputs\qa_downstream_task_explanation.md