"""
Evaluate neighborhood consistency of representations.

This metric measures whether nearest neighbors in the embedding space
share ontology-aligned concepts.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate neighborhood consistency."
    )
    ap.add_argument("--z_path", required=True)
    ap.add_argument("--z_dim", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)

    ap.add_argument("--labels_parquet", required=True)
    ap.add_argument("--k_neighbors", type=int, default=10)

    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    z = np.memmap(
        args.z_path,
        dtype=np.float32,
        mode="r",
        shape=(args.N, args.z_dim),
    )
    z = l2_normalize(np.asarray(z))

    labels = pd.read_parquet(args.labels_parquet)
    if not {"study_id", "concept_ids"}.issubset(labels.columns):
        raise KeyError("labels_parquet must contain [study_id, concept_ids].")

    # --------------------------------------------------
    # Similarity
    # --------------------------------------------------
    sim = cosine_similarity(z)
    np.fill_diagonal(sim, -1.0)

    scores = []
    for i in range(min(len(labels), args.N)):
        gt = set(labels.iloc[i].concept_ids)
        if not gt:
            continue

        nn_idx = np.argsort(sim[i])[-args.k_neighbors :]
        overlap = 0
        for j in nn_idx:
            other = set(labels.iloc[j].concept_ids)
            if gt & other:
                overlap += 1

        scores.append(overlap / args.k_neighbors)

    score = float(np.mean(scores)) if scores else 0.0

    out = {
        "metric": "neighborhood_consistency",
        "k_neighbors": args.k_neighbors,
        "value": score,
        "num_samples": len(scores),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[DONE] neighborhood consistency = {score:.4f}")


if __name__ == "__main__":
    main()
