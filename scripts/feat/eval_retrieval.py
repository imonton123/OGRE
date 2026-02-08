"""
Evaluate ontology concept retrieval performance.

Given image representations and ontology embeddings, this script
measures retrieval quality using recall@K.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_concept_index(csv_path: Path):
    df = pd.read_csv(csv_path)
    for col in ["concept_id", "cui", "node_id", "id"]:
        if col in df.columns:
            ids = df[col].astype(str).tolist()
            return {cid: i for i, cid in enumerate(ids)}, ids
    raise KeyError("No valid concept id column found.")


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate ontology concept retrieval (recall@K)."
    )
    ap.add_argument("--z_path", required=True)
    ap.add_argument("--z_dim", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)

    ap.add_argument("--kg_ids_csv", required=True)
    ap.add_argument("--kg_vecs_npy", required=True)

    ap.add_argument("--labels_parquet", required=True)
    ap.add_argument("--topk", type=int, default=10)

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

    concept_to_row, _ = load_concept_index(Path(args.kg_ids_csv))
    kg_vecs = np.load(args.kg_vecs_npy).astype(np.float32)
    kg_vecs = l2_normalize(kg_vecs)

    labels = pd.read_parquet(args.labels_parquet)
    if not {"study_id", "concept_ids"}.issubset(labels.columns):
        raise KeyError("labels_parquet must contain [study_id, concept_ids].")

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------
    recalls = []
    for i, row in labels.iterrows():
        if i >= args.N:
            break

        gt = [
            concept_to_row[c]
            for c in row.concept_ids
            if c in concept_to_row
        ]
        if not gt:
            continue

        sim = z[i] @ kg_vecs.T
        topk_idx = np.argpartition(sim, -args.topk)[-args.topk :]

        hit = any(g in topk_idx for g in gt)
        recalls.append(float(hit))

    recall_at_k = float(np.mean(recalls)) if recalls else 0.0

    out = {
        "metric": f"recall@{args.topk}",
        "value": recall_at_k,
        "num_samples": len(recalls),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[DONE] recall@{args.topk} = {recall_at_k:.4f}")


if __name__ == "__main__":
    main()
