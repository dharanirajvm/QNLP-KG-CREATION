# Explainable Reasoning Module Design

## Goal

Given a predicted triple or QA answer, produce a **supporting explanation** from the FB15k graph and the trained KGE model.

Example:

- query: `(Barack Obama, place_of_birth, ?)`
- predicted answer: `Honolulu`
- explanation:
  - short supporting paths from `Barack Obama` to `Honolulu`
  - shared-neighbor evidence
  - similar entities that also satisfy the same relation
  - explanation score and confidence

## Chosen algorithmic basis

We should not invent a new explanation method here. The most defensible and feasible design is:

### 1. Core path extractor: SFE / PRA-style local subgraph extraction

Use **Subgraph Feature Extraction (SFE)** with bounded BFS around the head and tail entities.

Why:

- proven KB completion / explanation-style feature extraction method
- more efficient and more expressive than plain PRA
- directly matches the need to find short typed paths and shared-neighbor patterns

What we use from it:

- local BFS around the entities
- typed path pattern extraction
- one-sided and shared-neighbor style features

### 2. Path reliability scorer: PCRA from PTransE

Use **Path-Constraint Resource Allocation (PCRA)** to score how reliable a candidate path is between the entities.

Why:

- directly designed for relation-path reliability in KGs
- proven and lightweight
- ideal for ranking short explanation paths

What we use from it:

- reliability of a path instance from `h` to `t`
- reliability of a path pattern for a predicted relation

### 3. Optional symbolic explainer layer: XKE-style post-hoc explanation

Use **XKE-style post-hoc explanation** later if we want a stronger research contribution.

Why:

- specifically proposed for explaining KGE predictions
- uses symbolic features / rules to explain opaque embedding decisions

What we use from it:

- relation-specific weighted path/rule explanations
- fidelity evaluation against the KGE model

## Final design choice

For **Version 1**, we should implement:

- **SFE-style local path extraction**
- **PCRA-style path reliability scoring**
- **embedding similarity support signals**

For **Version 2**, we can add:

- **XKE-style surrogate rule learning**

That is the right tradeoff between rigor and feasibility.

## Module scope

This module will explain outputs from:

1. KG completion
2. QA
3. semantic similarity

## Inputs

### Triple explanation mode

Input:

- head entity `h`
- relation `r`
- predicted or known tail `t`

### QA explanation mode

Input:

- question
- parsed KG query
- predicted answer entity

### Similarity explanation mode

Input:

- anchor entity `a`
- target entity `b`

## Output format

Each explanation should return:

- `query`
- `prediction`
- `prediction_score`
- `confidence_gap`
- `supporting_paths`
- `shared_neighbors`
- `similar_entity_support`
- `final_explanation_score`
- `natural_language_explanation`

## Core pipeline

### Step 1. Local subgraph extraction

Extract a bounded local subgraph around the involved entities.

Default:

- BFS depth = 2
- optional depth = 3 for debugging / rich explanations

For triple `(h, r, t)`:

- expand around `h`
- expand around `t`
- keep the union of those neighborhoods

This gives candidate evidence paths without searching the whole graph.

### Step 2. Candidate explanation generation

Generate three types of explanation evidence.

#### A. Short typed paths

Examples:

- `(h, r1, x) -> (x, r2, t)`
- `(h, r1, x) -> (x, r2, y) -> (y, r3, t)`

These are the main symbolic explanations.

#### B. Shared neighbors / overlap motifs

Examples:

- both `h` and `t` connect to the same entity `x`
- both connect to similar relation types

These act as structural support.

#### C. Similar entities with same relation

Examples:

- entities similar to `h` that also satisfy relation `r` with entities similar to `t`
- nearest embedding neighbors of `h`
- nearest embedding neighbors of `t`

These act as analogical support.

## Path scoring

We rank candidate explanations with a weighted score:

```text
explanation_score =
  w_path_rel * path_reliability
  + w_rel * relation_relevance
  + w_freq * path_frequency
  + w_emb * embedding_support
```

### 1. Path reliability

Use **PCRA-style reliability** for an instantiated path from `h` to `t`.

Interpretation:

- how much path flow actually reaches the target through this path

### 2. Relation relevance

Measure how predictive a path pattern is for the target relation.

Default estimator:

```text
relation_relevance(p, r) = count(path_pattern p supporting relation r) / count(path_pattern p)
```

This is simple, proven, and easy to compute on the training graph.

Later, Version 2 can replace this with learned rule weights from an XKE-style surrogate.

### 3. Path frequency

Measure how often the path pattern occurs for the target relation in training data.

Interpretation:

- rare paths may be noisy
- recurring paths are more trustworthy

### 4. Embedding support

Use cosine similarity between:

- `h` and similar heads that already satisfy relation `r`
- `t` and similar tails already used with relation `r`

Interpretation:

- symbolic path says the explanation is structurally plausible
- embedding support says it is also consistent with the learned KGE space

## Shared-neighbor scoring

For shared neighbor `x`, score using:

```text
shared_score =
  relation_match_strength
  + neighbor_popularity_penalty
  + embedding_coherence
```

Where:

- `relation_match_strength`: whether the incident relations are relevant to the target relation
- `neighbor_popularity_penalty`: penalize hubs
- `embedding_coherence`: cosine-based support from involved entities

## Similar-entity support

For relation `(h, r, t)`, collect:

- nearest neighbors of `h`
- nearest neighbors of `t`
- known triples `(h', r, t')` where:
  - `sim(h, h')` is high
  - `sim(t, t')` is high

This gives analogical explanations like:

- "Similar people also have the same birthplace relation pattern"

## What we will implement first

### Phase 1

Implement a deterministic, faithful explanation engine:

1. local subgraph extraction
2. path enumeration up to length 2
3. shared-neighbor extraction
4. similar-entity evidence
5. heuristic scoring with:
   - PCRA-style path reliability
   - path frequency
   - relation relevance
   - embedding support

This phase is enough for a strong demo and UI integration.

### Phase 2

Add a learned relation-specific explainer:

- build SFE feature vectors per relation
- train logistic regression / sparse linear model
- use learned weights as explanation relevance scores

### Phase 3

Add XKE-style post-hoc explanation:

- explain KGE predictions using weighted symbolic rules
- evaluate fidelity to the KGE model

## Why this design is strong

Because it combines:

- **symbolic evidence** from actual graph paths
- **reliability scoring** from established KG path work
- **embedding support** from the trained KGE

So the module is:

- defensible
- explainable
- feasible with your current codebase

## File plan

Recommended files:

- `LLM-Simplification/fQCE_V2/module_explainable_reasoning/README.md`
- `LLM-Simplification/fQCE_V2/module_explainable_reasoning/explanation_utils.py`
- `LLM-Simplification/fQCE_V2/module_explainable_reasoning/task4_explain_predictions_fb15k.py`
- `LLM-Simplification/fQCE_V2/module_explainable_reasoning/ui_explanation_helpers.py`

## UI plan

Add a new tab:

- `Explanation`

Modes:

1. explain KG completion
2. explain QA answer
3. explain semantic similarity

Show:

- predicted answer
- supporting paths table
- shared-neighbor table
- similar-entity evidence table
- explanation summary sentence

## Evaluation plan

We should evaluate:

### 1. Coverage

How many predictions get at least one explanation path.

### 2. Fidelity

How well explanation features agree with the KGE prediction ranking.

### 3. Plausibility

Manual inspection of top explanations.

### 4. Efficiency

Average explanation runtime per query.

## Practical recommendation

Start with:

- path length <= 2
- top 20 local neighbors per expansion
- top 10 explanations returned

That is enough to keep the module fast and presentable.

## References to base the design on

- PRA: path-constrained random walk style inference
- SFE: efficient local subgraph feature extraction for KB completion
- PTransE: path reliability using PCRA
- XKE: post-hoc symbolic explanation for KGE models

## Decision

We will build this module as:

**SFE-style path extraction + PCRA-style path scoring + KGE embedding support**, with **XKE-style surrogate explanations** as a later extension.
