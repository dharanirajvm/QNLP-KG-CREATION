# Full Project Context

## Project Title

**End-to-End QNLP-Driven Knowledge Graph Construction, Embedding Learning, Downstream Reasoning, and Explainability**

## Project Overview

This project develops a unified pipeline that begins with **Quantum Natural Language Processing (QNLP)** and ends with **interpretable knowledge-driven inference**. The overall goal is to convert natural language into structured relational knowledge using a **DisCoCat/Lambeq compositional semantic framework**, transform those semantics into a knowledge graph, learn embeddings over the resulting triples, apply those embeddings to downstream reasoning tasks, and finally improve trust in the system through an explainability layer.

The work is motivated by the observation that natural language contains rich semantic and relational structure, but that structure is inherently unstructured and difficult to convert directly into machine-readable knowledge. Classical knowledge graph generation and embedding methods often ignore grammatical compositionality and semantic roles, while current QNLP work frequently focuses on isolated tasks rather than complete end-to-end pipelines. This project addresses that gap by connecting **QNLP-based knowledge graph creation**, **knowledge graph embedding generation**, **downstream reasoning**, and **explainable decision support** within a single coherent framework.

## Core Research Problem

The central problem addressed in this project is:

> How can a grammar-aware QNLP pipeline be used to construct knowledge graphs from language, learn meaningful relational representations over those graphs, apply those representations to downstream reasoning tasks, and provide interpretable evidence for the resulting predictions?

## End-to-End Workflow

The pipeline is organized into four major stages:

1. **Knowledge Graph Creation from QNLP**
2. **Knowledge Graph Embedding Learning**
3. **Downstream Reasoning Tasks**
4. **Explainability and Interpretation**

Each stage is described below.

## 1. Knowledge Graph Creation from QNLP

The first part of the project focuses on generating knowledge graph triples from text using a **DisCoCat/Lambeq-based compositional semantic pipeline**. This stage is important because it preserves grammatical structure and semantic roles rather than relying only on flat vector representations.

The objective of this stage is to:

- process natural language using a compositional semantic framework
- extract structured entity-relation-entity triples
- build a machine-readable knowledge graph from linguistically grounded semantics

This part of the work is represented in:

- [module1_kge/README.md](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\README.md)
- [run_module1_kge_pipeline.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\run_module1_kge_pipeline.py)

The `module1_kge` pipeline supports:

- building dataset splits from Lambeq-generated triples and/or dataset CSV files
- preparing `train.txt`, `valid.txt`, and `test.txt`
- deduplication and confidence filtering
- training and analysis on the generated graph

This stage establishes the QNLP-driven graph construction foundation of the project.

## 2. Knowledge Graph Embedding Learning

After triples are generated, the next stage is to learn embeddings that capture the relational structure of the graph. This part of the project studies knowledge graph embedding learning in both a smaller QNLP-generated setting and a larger benchmark setting.

Two main tracks were developed:

### 2.1 Module-1 KGE pipeline for DisCoCat/Lambeq triples

This pipeline was used to:

- train a KGE model on the toy or task-specific graph derived from the DisCoCat/Lambeq pipeline
- visualize embeddings
- analyze the semantic meaning of learned embeddings

Relevant files:

- [run_module1_kge_pipeline.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\run_module1_kge_pipeline.py)
- [visualize_module1_embeddings.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\visualize_module1_embeddings.py)
- [analyze_module1_kge_meaning.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\analyze_module1_kge_meaning.py)

### 2.2 Large-scale benchmark KGE on FB15k-237

To demonstrate stronger relational learning and more credible downstream inference, the project also uses a benchmark knowledge graph setting based on **FB15k-237**.

This stage includes:

- training on a larger benchmark graph
- improving training with **relation-negative sampling**
- logging training in detail
- saving a strong snapshot for inference and downstream use

Relevant files:

- [training_fb15k237.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-2 outputs\code\training_fb15k237.py)
- [training_fb15k237_v2.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\training_fb15k237_v2.py)
- [fQCE_V2/README.md](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\README.md)

The V2 work strengthens the model by:

- corrupting both entities and relations during training
- making the model more robust for relation prediction
- storing detailed epoch and batch logs in `training.log`

The primary stable inference snapshot used in later modules is:

- `LLM-Simplification/fQCE/inference_snapshots/quantum_fb15k237_20260308_174529_updated_20260310_193344`

This larger model serves as the main backbone for downstream tasks and explainability.

## 3. Downstream Reasoning Tasks

Once embeddings are learned, the next part of the project is to show that they are useful beyond training metrics. This is done through downstream reasoning tasks.

Three downstream tasks were implemented:

### 3.1 Knowledge Graph Completion

This task extends link prediction and predicts missing parts of triples such as:

- `(h, r, ?)`
- `(?, r, t)`
- `(h, ?, t)`

This demonstrates that the learned embeddings encode relational structure well enough to recover missing graph facts.

Relevant files:

- [task1_kg_completion.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\task1_kg_completion.py)
- [task1_kg_completion_fb15k.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\task1_kg_completion_fb15k.py)

### 3.2 Semantic Similarity

This task uses learned entity embeddings to compute semantic similarity, typically with cosine similarity between two entity embeddings or by ranking nearest entities in embedding space.

This stage demonstrates whether the learned embedding space captures semantic proximity between entities.

Relevant files:

- [task2_semantic_retrieval.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\task2_semantic_retrieval.py)
- [task2_semantic_retrieval_fb15k.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\task2_semantic_retrieval_fb15k.py)
- [semantic_similarity_task_explanation.md](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-2 outputs\semantic_similarity_task_explanation.md)

### 3.3 Question Answering

This task converts a natural-language question into a graph query and uses graph evidence plus learned embeddings to retrieve an answer.

Typical examples include:

- “Where was Barack Obama born?”
- “Where does Emma study?”

Relevant files:

- [task3_qa_answering.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\task3_qa_answering.py)
- [task3_qa_answering_fb15k.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\task3_qa_answering_fb15k.py)

Together, these downstream tasks show that the learned representations are useful for:

- inference over incomplete graphs
- semantic reasoning in embedding space
- natural-language question answering through graph-based retrieval

## 4. Explainability and Interpretation

A major motivation for the later phase of the project is that embedding-based systems can make plausible predictions without clearly explaining why. To address this, an explainability layer was developed on top of the learned knowledge graph embeddings.

The explainability module aims to answer:

- Why was this triple predicted?
- What graph evidence supports this answer?
- Which structural or semantic signals contribute to the prediction?

The current explainability framework combines three evidence sources:

1. **Supporting paths**
2. **Shared-neighbor evidence**
3. **Analogical support from the embedding space**

### 4.1 Supporting paths

These are short symbolic paths such as:

- `(h, r1, x) -> (x, r2, t)`

These paths provide interpretable structural evidence from the graph.

### 4.2 Shared-neighbor evidence

These explanations identify intermediate entities that connect both the head and tail and may act as structural bridges.

### 4.3 Analogical embedding support

This uses the learned embedding space to identify similar entities or similar known triples that support the current prediction by analogy.

Relevant files:

- [explanation_utils.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\code\explanation_utils.py)
- [task4_explain_predictions_fb15k.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\code\task4_explain_predictions_fb15k.py)
- [explainable_reasoning_module_design.md](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\explainable_reasoning_module_design.md)

The explainability stage is grounded in ideas inspired by:

- PRA and SFE for path-based reasoning
- PTransE/PCRA for path reliability intuition
- embedding-based analogical support for latent semantic evidence

## Learned Explanation Weights

To move beyond purely heuristic explanation scoring, a learned weighting mechanism was developed.

This stage trains separate models for:

- path explanation scoring
- shared-neighbor explanation scoring

The learned models are stored in:

- `LLM-Simplification/Review-3 outputs/models/learned_explanation_weights/`

Relevant file:

- [train_explanation_weights.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\code\train_explanation_weights.py)

Key properties of this training:

- training is performed on **GPU only**
- detailed run logs are stored in `training.log`
- the learned coefficients are **sign-constrained** to preserve interpretability

The constraints ensure that:

- supportive features retain nonnegative influence where appropriate
- hubness is penalized through a nonpositive hubness weight

This improves the semantic plausibility of the learned explanation scorer.

## Dataset Exploration and Visualization

To better understand the benchmark graph and choose strong explainability examples, dataset visualization tools were also developed.

Relevant files:

- [visualize_fb15k_dataset.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\code\visualize_fb15k_dataset.py)
- [visualize_fb15k_full_kg.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\code\visualize_fb15k_full_kg.py)

These support:

- local subgraph exploration
- triple-centered subgraph inspection
- relation-level graph exploration
- full-graph export to GraphML and GEXF

This stage helps connect raw graph structure to downstream reasoning and explainability choices.

## User Interface Integration

To make the system easier to inspect and demonstrate, the project also includes a UI layer that integrates:

- inference
- downstream tasks
- embeddings
- metrics
- explainability

Relevant UI files:

- [review2_kge_ui.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-2 outputs\code\review2_kge_ui.py)
- [review2_kge_ui.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\review2_kge_ui.py)

The UI now supports:

- KGE inference
- downstream task interaction
- explainability tab with path graphs and context subgraphs
- tables for supporting paths, shared neighbors, and analogical support

## Main Contribution of the Project

The full contribution of the project is not only the development of a knowledge graph embedding model, but the creation of a complete research pipeline that connects:

- **QNLP-based semantic graph construction**
- **knowledge graph embedding learning**
- **downstream relational reasoning**
- **explainable graph-based inference**

This makes the project broader than a standard embedding benchmark study. It contributes to the intersection of:

- Quantum Natural Language Processing
- knowledge graph construction
- relational learning
- downstream semantic inference
- explainable AI for graph representations

## Thesis-Oriented Summary

In thesis terms, this project studies how a grammar-aware QNLP pipeline can be used to construct knowledge graphs from language, how meaningful representations can be learned from those graphs, how those representations can support downstream inference tasks, and how the resulting predictions can be explained in a structurally and semantically interpretable way.

The project therefore combines:

- **QNLP and compositional semantics**
- **knowledge graph creation**
- **embedding-based reasoning**
- **benchmark evaluation**
- **explainability**

within one end-to-end research framework.

## Suggested Thesis Framing

The project can be described as:

> A unified framework for QNLP-driven knowledge graph creation, embedding learning, downstream reasoning, and explainable relational inference.

## Key File Map

### QNLP and KG creation

- [run_module1_kge_pipeline.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\run_module1_kge_pipeline.py)
- [module1_kge/README.md](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE\module1_kge\README.md)

### KGE training

- [training_fb15k237.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-2 outputs\code\training_fb15k237.py)
- [training_fb15k237_v2.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\training_fb15k237_v2.py)
- [fQCE_V2/README.md](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\README.md)

### Downstream tasks

- [task1_kg_completion_fb15k.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\task1_kg_completion_fb15k.py)
- [task2_semantic_retrieval_fb15k.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\task2_semantic_retrieval_fb15k.py)
- [task3_qa_answering_fb15k.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\fQCE_V2\task3_qa_answering_fb15k.py)

### Explainability

- [explanation_utils.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\code\explanation_utils.py)
- [task4_explain_predictions_fb15k.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\code\task4_explain_predictions_fb15k.py)
- [train_explanation_weights.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\code\train_explanation_weights.py)
- [explainable_reasoning_module_design.md](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-3 outputs\explainable_reasoning_module_design.md)

### UI

- [review2_kge_ui.py](C:\Users\DHARANIRAJ VM\Documents\FYP-25\QNLP\Phase-1\lambeq\qnlp\LLM-Simplification\Review-2 outputs\code\review2_kge_ui.py)

