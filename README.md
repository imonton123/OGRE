# OGRE: Ontology-Guided Representation Engineering

This repository provides the **official implementation** of **OGRE (Ontology-Guided Representation Engineering)**,  
a framework for learning **ontology-aligned, sparse, and interpretable representations** for medical vision-language models.

OGRE integrates structured medical knowledge (UMLS ontology) into representation learning to improve **robustness, interpretability, and semantic consistency**.

---

## 🔍 Overview

<p align="center">
  <img src="asserts/Fig_1.png" width="95%">
</p>

**Figure 1.** Overview of the OGRE framework.  
OGRE constructs a lightweight medical ontology from UMLS, extracts ontology-aligned concept prototypes, and trains representations that are explicitly guided by ontology structure.

---

## ✨ Key Contributions

- **Ontology-Guided Representation Learning**  
  Explicitly aligns visual representations with structured medical knowledge.

- **Sparse and Interpretable Latent Space**  
  Learns sparse latent dimensions corresponding to semantically meaningful concepts.

- **Improved Robustness and Semantic Consistency**  
  Encourages stable representation geometry under dataset shifts and perturbations.

---

## 🧱 Ontology Construction

We construct a lightweight medical ontology from **UMLS** using a transparent and reproducible pipeline:

### 1. Node Construction
- Select preferred English concept names.
- Attach semantic types (TUIs) to each concept.

### 2. Semantic & Lexical Filtering
- Retain clinically meaningful semantic types.
- Remove ambiguous or non-clinical concepts using lexical constraints.

### 3. Edge Construction
- Build a directed ontology graph using hierarchical relations  
  (e.g., `isa`, `part_of`).

The ontology construction pipeline is implemented in:

```text
scripts/kg/
├── build_umls_nodes.py     # UMLS → ontology nodes
├── filter_nodes.py         # semantic & lexical filtering
└── build_umls_edges.py     # hierarchical relations
```

---

## 🧠 Representation Learning Pipeline
OGRE learns ontology-aligned representations in three stages:

### 1️⃣ Sparse Transcoder Training
Learns a sparse latent representation from image embeddings.
```text
scripts/feat/train_transcoder.py
```

### 2️⃣ Ontology Prototype Construction
Builds ontology prototype vectors by aggregating aligned ontology concepts.
```text
scripts/feat/build_proto_cache.py
```

### 3️⃣ Ontology-Guided Representation Training (GRT)
Aligns image representations with ontology prototypes.
```text
scripts/feat/train_grt.py
```

----
## 📊 Evaluation

We evaluate representation quality using ontology-aware metrics:
- Concept Retrieval (Recall@K)
    Measures whether representations retrieve correct ontology concepts.

- Neighborhood Consistency
    Evaluates semantic consistency in representation space.

Evaluation scripts are provided in:
```text
scripts/feat/
├── eval_retrieval.py
└── eval_neighborhood.py
```

---

## 📁 Repository Structure
```text
.
├── scripts/
│   ├── kg/        # ontology construction
│   ├── feat/      # representation learning & evaluation
│   └── utils/     # shared utilities
├── asserts/
│   └── Fig_1.png
└── README.md
```

---

## 📄 Dataset & Preprocessing
Due to data privacy constraints, **raw datasets and dataset-specific preprocessing scripts are not publicly released.**
- Ontology concepts are extracted using QuickUMLS.
- Image–concept linking and dataset-specific filtering are described in the paper.

---
## 🧑‍💻 Notes
- This repository focuses on **methodological clarity and reproducibility.**
- Some implementation details are intentionally omitted to respect data usage agreements.