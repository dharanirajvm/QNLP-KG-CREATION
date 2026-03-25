# Semantic Similarity Task: Correct KGE Formulation

## What semantic similarity should mean here

For a KGE model, semantic similarity is an **embedding-space comparison between entities that already live in the same learned vector space**.

That means:

- take the embedding of entity `e1`
- take the embedding of entity `e2`
- normalize both
- compute **cosine similarity**

This is the correct operation for questions like:

- "How similar are `Barack Obama` and `Joe Biden` in the learned embedding space?"
- "What are the nearest entities to `Harvard University`?"

## What was wrong before

The previous implementation was doing **hybrid text retrieval**:

- build TF-IDF documents from graph neighborhoods
- compare a free-text query against those documents
- optionally rerank with KGE scores

That can be a retrieval baseline, but it is **not** semantic similarity over KGE embeddings.

The core issue is simple:

- cosine similarity is valid only when both inputs are represented in the **same embedding space**
- a free-text query like `"american universities"` was not embedded by the KGE model
- so the old code mixed a text-space score with a graph-embedding score and called it semantic similarity

That is the wrong formulation for this task.

## Correct implementation

The fixed implementation now does two things:

### 1. Pairwise similarity

Given:

- anchor entity `a`
- target entity `b`

we compute:

```python
sim(a, b) = cosine(emb(a), emb(b))
```

Since entity embeddings are L2-normalized, this is just the dot product:

```python
sim(a, b) = emb(a) · emb(b)
```

### 2. Nearest-neighbor similarity search

Given an anchor entity `a`, we compute cosine similarity against all entity embeddings:

```python
score_i = cosine(emb(a), emb(entity_i))
```

Then we rank all entities by descending cosine similarity and return the top neighbors.

This is the standard semantic-similarity / nearest-neighbor setup for embeddings.

## Why this is the right approach

Because the KGE model learns one embedding space for entities, cosine similarity in that space directly captures which entities are placed near each other by the model.

So if the model is meaningful:

- similar people should be near similar people
- similar locations near similar locations
- similar institutions near similar institutions

This is exactly what nearest-neighbor analysis in embedding research usually does.

## Important limitation

This approach works for:

- entity-to-entity similarity
- nearest-neighbor retrieval from an anchor entity

It does **not** directly solve:

- free-text semantic search like `"universities in USA"`

For that, you need one of these:

1. a text encoder that maps the query into the same space
2. a separate retrieval layer over labels/descriptions
3. a structured KG query derived from text

So:

- **semantic similarity** = cosine over KGE entity embeddings
- **semantic search from raw text** = different task, needs additional machinery

## What is implemented now

The corrected code now supports:

- anchor entity -> nearest similar entities
- anchor entity + target entity -> pairwise cosine similarity

Files updated:

- `LLM-Simplification/fQCE/module1_kge/downstream_utils.py`
- `LLM-Simplification/fQCE/module1_kge/task2_semantic_retrieval.py`
- `LLM-Simplification/fQCE_V2/task2_semantic_retrieval_fb15k.py`
- `LLM-Simplification/fQCE/review2_kge_ui.py`
- `LLM-Simplification/Review-2 outputs/code/review2_kge_ui.py`

## Example

```powershell
python "LLM-Simplification/fQCE_V2/task2_semantic_retrieval_fb15k.py" `
  --anchor /m/02mjmr `
  --target /m/0d06m5 `
  --top-k 5
```

This will:

- compute cosine similarity between the two selected FB15k entities
- return the nearest neighbors of the anchor entity

## Bottom line

Yes, in general semantic similarity between two embeddings is computed using cosine similarity.

For your KGE setting, the correct task is:

- compare **entity embeddings to entity embeddings**
- not **raw text queries to entity embeddings**
