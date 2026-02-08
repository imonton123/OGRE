"""
Build ontology prototype vectors for each sample.

For each image embedding z_i, this script computes similarity scores
against ontology (KG) embeddings and caches top-aligned ontology
prototypes for downstream training and evaluation.

Input:
  --z_path          : memmap (.npy) of image embeddings [N, D]
  --row_idx_path    : memmap (.npy) mapping rows to study_id
  --feat_ids_csv    : CSV with column [study_id]
  --kg_ids_csv      : CSV defining ontology concept IDs
  --kg_vecs_npy     : numpy array of KG embeddings [M, D]

Output:
  --out_npy         : numpy array of prototype vectors [N, D]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_kg_index(kg_ids_csv: Path):
    df = pd.read_csv(kg_ids_csv)
    for col in ["concept_id", "cui", "node_id", "id"]:
        if col in df.columns:
            ids = df[col].astype(str).tolist()
            return {cid: i for i, cid in enumerate(ids)}, ids
    raise KeyError(f"No valid concept ID column found in {kg_ids_csv}")


# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build ontology prototype cache for each sample."
    )
    ap.add_argument("--z_path", required=True, type=str)
    ap.add_argument("--row_idx_path", required=True, type=str)
    ap.add_argument("--feat_ids_csv", required=True, type=str)

    ap.add_argument("--kg_ids_csv", required=True, type=str)
    ap.add_argument("--kg_vecs_npy", required=True, type=str)

    ap.add_argument("--out_npy", required=True, type=str)

    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--z_dim", type=int, required=True)
    ap.add_argument("--topk", type=int, default=32)

    args = ap.parse_args()

    # --------------------------------------------------
    # Load feature metadata
    # --------------------------------------------------
    ids_df = pd.read_csv(args.feat_ids_csv)
    if "study_id" not in ids_df.columns:
        raise KeyError("feat_ids_csv must contain column 'study_id'")
    ids_df["study_id"] = ids_df["study_id"].astype(str)

    row_idx = np.memmap(
        args.row_idx_path, dtype=np.int64, mode="r", shape=(args.N,)
    )
    z = np.memmap(
        args.z_path, dtype=np.float32, mode="r", shape=(args.N, args.z_dim)
    )

    # --------------------------------------------------
    # Load KG embeddings
    # --------------------------------------------------
    concept_to_row, _ = load_kg_index(Path(args.kg_ids_csv))
    kg_vecs = np.load(args.kg_vecs_npy).astype(np.float32)
    kg_vecs = l2_normalize(kg_vecs)

    # --------------------------------------------------
    # Build prototype cache
    # --------------------------------------------------
    out_path = Path(args.out_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    proto_cache = np.memmap(
        out_path,
        dtype=np.float32,
        mode="w+",
        shape=(args.N, args.z_dim),
    )

    for i in range(args.N):
        z_i = z[i : i + 1]
        z_i = l2_normalize(z_i)

        sim = z_i @ kg_vecs.T
        idx = np.argpartition(sim[0], -args.topk)[-args.topk :]
        proto = kg_vecs[idx].mean(axis=0, keepdims=True)
        proto_cache[i] = proto.astype(np.float32)

        if i % 2000 == 0 or i + 1 == args.N:
            print(f"[proto] {i+1}/{args.N}")

    proto_cache.flush()
    print(f"[DONE] saved prototype cache: {out_path}")


if __name__ == "__main__":
    main()
