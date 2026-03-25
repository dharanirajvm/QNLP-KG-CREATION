# fQCE_V2

This folder contains the V2 KGE trainer with:

- relation-negative sampling in the quantum training loop
- filtered relation-prediction evaluation on validation and test splits
- outputs saved inside this V2 folder by default
- detailed step-by-step run logging written to `training.log`

## Run

```powershell
python "LLM-Simplification/fQCE_V2/training_fb15k237_v2.py" `
  --dataset-dir "LLM-Simplification/fQCE/module1_kge/datasets/discocat_lambeq_20260311_194819" `
  --model quantum `
  --q-backend default.qubit `
  --epochs 40 `
  --entity-negatives-per-positive 1 `
  --relation-negatives-per-positive 1 `
  --train-log-every-batches 10 `
  --log-first-n-batches 3
```

## Default outputs

- Runs: `LLM-Simplification/fQCE_V2/runs_kge_v2/`
- Each run saves `config.json`, `entity_to_id.json`, `relation_to_id.json`, `best_model.pt`, `last_model.pt`, `metrics_history.jsonl`, `metrics_summary.json`, `training.log`

## Logging controls

- `--train-log-every-batches`: periodic detailed batch logging
- `--log-first-n-batches`: always log the first few batches of each epoch
- `--log-sampled-triples-per-epoch`: log sampled training triples at epoch start

## FB15k-237 Downstream Tasks

These downstream scripts default to the strong FB15k-237 snapshot:

- `LLM-Simplification/fQCE/inference_snapshots/quantum_fb15k237_20260308_174529_updated_20260310_193344`

### 1. KG Completion

```powershell
python "LLM-Simplification/fQCE_V2/task1_kg_completion_fb15k.py" `
  --mode tail `
  --head /m/02mjmr `
  --relation /people/person/place_of_birth `
  --top-k 5
```

### 2. Semantic Similarity

```powershell
python "LLM-Simplification/fQCE_V2/task2_semantic_retrieval_fb15k.py" `
  --anchor /m/02mjmr `
  --target /m/0d06m5 `
  --top-k 5
```

### 3. QA

```powershell
python "LLM-Simplification/fQCE_V2/task3_qa_answering_fb15k.py" `
  --question "Where was Barack Obama born?" `
  --top-k 5
```
