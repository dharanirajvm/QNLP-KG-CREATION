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
