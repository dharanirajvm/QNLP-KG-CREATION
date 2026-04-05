# Review-3 Outputs

This folder contains the Phase 1 implementation of the **Explainable Reasoning Module** for the FB15k-237 quantum KGE model.

## Default model used

All code here defaults to the strong FB15k snapshot trained for the larger run you have been using:

- `LLM-Simplification/fQCE/inference_snapshots/quantum_fb15k237_20260308_174529_updated_20260310_193344`

and the FB15k dataset:

- `LLM-Simplification/fQCE/datasets/fb15k237`

Those paths are already hard-wired as defaults in the code.

## Phase 1 implemented

Phase 1 explains a predicted or provided triple using:

- local subgraph extraction around the involved entities
- short supporting paths of the form `(h, r1, x) -> (x, r2, t)`
- shared-neighbor evidence
- similar-entity analogical support from the trained KGE space
- weighted explanation ranking using:
  - path reliability
  - relation relevance
  - path frequency
  - embedding support

## Files

- `code/explanation_utils.py`
- `code/task4_explain_predictions_fb15k.py`
- `code/visualize_fb15k_dataset.py`
- `code/visualize_fb15k_full_kg.py`

## FB15k Dataset Explorer

This explorer helps us inspect the FB15k graph first, so we can choose better examples for the explainability layer.

Generate an overall dataset summary:

```powershell
python "LLM-Simplification/Review-3 outputs/code/visualize_fb15k_dataset.py" `
  --mode overview
```

Visualize the local neighborhood around an entity:

```powershell
python "LLM-Simplification/Review-3 outputs/code/visualize_fb15k_dataset.py" `
  --mode entity `
  --entity "Barack Obama"
```

Visualize a triple-centered local subgraph:

```powershell
python "LLM-Simplification/Review-3 outputs/code/visualize_fb15k_dataset.py" `
  --mode triple `
  --head "Barack Obama" `
  --relation "place of birth" `
  --tail "Honolulu"
```

Inspect a relation slice:

```powershell
python "LLM-Simplification/Review-3 outputs/code/visualize_fb15k_dataset.py" `
  --mode relation `
  --relation "profession"
```

Outputs are saved under:

- `LLM-Simplification/Review-3 outputs/outputs/fb15k_explorer/`

## Full Dataset KG Export

If we want the entire FB15k dataset as one graph export, use:

```powershell
python "LLM-Simplification/Review-3 outputs/code/visualize_fb15k_full_kg.py" `
  --write-graphml `
  --write-gexf
```

This writes:

- `nodes.csv`
- `edges.csv`
- `graph_summary.json`
- `fb15k_full.graphml`
- `fb15k_full.gexf`

under:

- `LLM-Simplification/Review-3 outputs/outputs/fb15k_full_kg/`

For the full FB15k graph, `GraphML` or `GEXF` is the better format to open in Gephi or Cytoscape. A browser-based interactive view is usually too heavy for the complete graph.

## Run

Explain a provided triple using text:

```powershell
python "LLM-Simplification/Review-3 outputs/code/task4_explain_predictions_fb15k.py" `
  --head "Barack Obama" `
  --relation "place of birth" `
  --tail "Honolulu"
```

Explain the top predicted tail for a query using text:

```powershell
python "LLM-Simplification/Review-3 outputs/code/task4_explain_predictions_fb15k.py" `
  --head "Barack Obama" `
  --relation "place of birth"
```

Explain a known training triple using text:

```powershell
python "LLM-Simplification/Review-3 outputs/code/task4_explain_predictions_fb15k.py" `
  --head "Barack Obama" `
  --relation "profession" `
  --tail "Lawyer"
```

Explain using raw FB15k IDs:

```powershell
python "LLM-Simplification/Review-3 outputs/code/task4_explain_predictions_fb15k.py" `
  --head /m/02mjmr `
  --relation /people/person/place_of_birth `
  --tail /m/02hrh0_
```

Save JSON output:

```powershell
python "LLM-Simplification/Review-3 outputs/code/task4_explain_predictions_fb15k.py" `
  --head "Barack Obama" `
  --relation "place of birth" `
  --output-json "LLM-Simplification/Review-3 outputs/explanation_sample.json"
```

## Design basis

This Phase 1 module follows the design:

- SFE-style local path extraction
- PTransE/PCRA-style path reliability intuition
- KGE embedding support from the trained model

The heavier XKE-style surrogate explanation layer is intentionally left for a later phase.

## Learn explanation weights from data

We now support a stronger Option B setup where the explainer learns scoring weights from FB15k data instead of relying only on fixed heuristics.

This trains separate logistic-regression models for:

- path explanations
- shared-neighbor explanations

The learned models are now sign-constrained so the coefficients stay semantically meaningful:

- path features are constrained to have nonnegative weights
- `relation_match_strength` and `embedding_coherence` are constrained to be nonnegative
- shared-neighbor hubness is modeled with `hubness_log` and constrained to have a nonpositive weight

Run:

```powershell
python "LLM-Simplification/Review-3 outputs/code/train_explanation_weights.py" `
  --device cuda `
  --max-positive-triples 4000 `
  --negatives-per-positive 2 `
  --epochs 400
```

The learned weights are saved under:

- `LLM-Simplification/Review-3 outputs/models/learned_explanation_weights/`

Once these files exist, the explainability CLI and UI load them automatically and use learned scores.

Training is GPU-only in this script. The run also writes a detailed log to:

- `LLM-Simplification/Review-3 outputs/models/learned_explanation_weights/training.log`
