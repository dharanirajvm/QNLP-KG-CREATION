# module1_kge

End-to-end fQCE pipeline for your DisCoCat/Lambeq triples:

1. Build KGE splits (`train.txt`, `valid.txt`, `test.txt`) from Lambeq triples and/or dataset CSV.
2. Train fQCE (`training_fb15k237.py`).
3. Generate training metric plots.
4. Generate embedding visualizations (PCA + t-SNE).
5. Run embedding meaning analysis.

## Run (full pipeline)

```powershell
python "LLM-Simplification/fQCE/module1_kge/run_module1_kge_pipeline.py" `
  --source-mode both `
  --deduplicate `
  --model quantum `
  --epochs 40 `
  --train-samples-per-epoch 0 `
  --q-backend default.qubit
```

## Common options

- `--source-mode lambeq_generated|dataset_csv|both`
- `--lambeq-triples-jsonl <path1> <path2> ...`
- `--dataset-csv <path1> <path2> ...`
- `--min-confidence 0.5` (filters records with confidence field)
- `--max-triples 0` (0 = use all)
- `--skip-training --existing-run-dir <run_dir>` (re-run only analysis/viz)
- `--skip-visualization`, `--skip-meaning-analysis`, `--skip-metrics-viz`

## Outputs

- Prepared dataset: `LLM-Simplification/fQCE/module1_kge/datasets/<dataset_tag>_<timestamp>/`
- Training run: `LLM-Simplification/fQCE/module1_kge/runs/<model>_fb15k237_<timestamp>/`
- Pipeline summary: `pipeline_summary_module1.json` inside the run directory.

## Downstream Tasks

### 1. Knowledge Graph Completion

```powershell
python "LLM-Simplification/fQCE/module1_kge/task1_kg_completion.py" `
  --snapshot-dir "LLM-Simplification/fQCE/module1_kge/runs/<run_dir>" `
  --mode tail `
  --head emma `
  --relation studies_at `
  --top-k 5
```

### 2. Semantic Retrieval

```powershell
python "LLM-Simplification/fQCE/module1_kge/task2_semantic_retrieval.py" `
  --snapshot-dir "LLM-Simplification/fQCE/module1_kge/runs/<run_dir>" `
  --query "students at yale" `
  --top-k 5
```

### 3. QA Answering

```powershell
python "LLM-Simplification/fQCE/module1_kge/task3_qa_answering.py" `
  --snapshot-dir "LLM-Simplification/fQCE/module1_kge/runs/<run_dir>" `
  --question "Where does emma study?" `
  --top-k 5
```
